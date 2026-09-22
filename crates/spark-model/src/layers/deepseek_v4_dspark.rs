// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 **native DSpark** block draft proposer (3 stages, γ = 5).
//!
//! Unlike every other proposer in the tree, DSpark does not run its block
//! autoregressively: one forward produces γ drafts at once, from γ embedded
//! tokens `[last_token, noise × (γ-1)]` that attend to each other AND to a
//! ring of the TARGET's own captured hidden states.
//!
//! ```text
//!  target step            for i, layer in layers:            (model/impl_b3.rs)
//!                             if i in dspark_target_layer_ids:
//!                                 capture[row][slot] = mean_over_hc(h)   ← dspark_capture.cu
//!
//!  propose()  ring        main_x = main_norm(main_proj(capture))
//!             catch-up    ring[s][pos % W] = rope(kv_norm(wkv_s(main_x)))
//!
//!             block       x = hc_expand(embed([last, noise×4]))
//!             stage s     hc_pre → attn_norm → MLA(q@pos..pos+γ-1, K=[ring‖block])
//!                         → hc_post → hc_pre → ffn_norm → MoE → hc_post
//!
//!             head        x = hc_head(h);  logits = lm_head(norm(x))
//!                         greedy Markov chain → γ drafts;  confidence from PRE-norm x
//! ```
//!
//! The submodules split the 500-LoC cap along that structure:
//! [`ring`] (pure host index algebra), [`capture`] (the capture kernel and
//! band bookkeeping), [`attn`] (one stage's MLA), [`stage`] (one stage's mHC
//! sites), [`propose`] (the `DraftProposer` impl and the ring catch-up), and
//! [`markov_confidence`] (the head, the chain and its host reference).
//!
//! ## What is NOT reused, and why
//!
//! The stage bodies are loaded through the same `assemble_layer_concrete` the
//! target layers use, but their FORWARD is re-implemented here.
//! `TransformerLayer::decode()` is one token against a paged causal cache and
//! `prefill()` is chunk self-attention that writes that cache; DSpark is γ
//! mutually-visible queries against a 128-slot ring of MAIN-MODEL KV with no
//! paged cache at all. The reference agrees: `get_dspark_topk_idxs` returns ONE
//! index list `.expand(bsz, block_size, -1)` for every query.
//!
//! ## EP=2
//!
//! [`DraftProposer::needs_comm`] is true, so both ranks load an EP-SHARDED
//! copy of the stages and both run the propose through the existing
//! `EP_CMD_MTP_PROPOSE` lane — no new protocol. The corollary is in
//! `model/impl_b3.rs`: the DSpark capture branch must NOT take the DFlash
//! rank-0 early return, or the worker would propose from a stale capture and
//! the two ranks would all-reduce partials of different inputs.

use std::any::Any;
use std::sync::atomic::AtomicU32;

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::qwen3_attention::{HcWeights, MlaWeights};
use crate::speculative::ProposerState;
use crate::weight_loader::deepseek_v4::{DSPARK_STAGES, DeepseekV4DsparkWeights};
use crate::weight_map::DenseWeight;

pub mod attn;
pub mod attn_proj;
pub mod capture;
pub mod markov_confidence;
pub mod propose;
pub mod ring;
pub mod stage;

#[cfg(test)]
mod ring_contract_tests;
#[cfg(test)]
mod tests;

pub use capture::{
    band_row, launch_hc_mean_capture, row_stride_elems, rows_through, slot_offset_elems,
};
pub use markov_confidence::{ChainOut, MarkovChainRef};

/// The largest γ this head will ever be asked for. The checkpoint's trained
/// block size is 5 and `attn::ATTN_BLOCK_ROWS` is 16, so nothing between them
/// needs a second code path.
pub const MAX_BLOCK: usize = 8;

/// Per-sequence DSpark state: the prefill capture ring and the three stage KV
/// rings, all device-owned (`DevicePtr` has no `Drop`, so `free_state` must
/// release them).
pub struct DsparkProposerState {
    /// `[window, n_targets * hidden]` BF16 — the last ≤ `window` prompt
    /// captures, at slot `pos % window`, written during prefill and drained on
    /// the FIRST propose (never from an eager end-of-prefill hook; see
    /// `prefill_uses_shared_buffers`).
    pub prefill_capture_ring: DevicePtr,
    /// How many prompt positions the prefill wrote (0 on a warm turn whose
    /// prefill was skipped — the gap path).
    pub prefill_filled: usize,
    /// The prompt length the prefill ring belongs to, so the drain knows which
    /// absolute positions its slots hold.
    pub prefill_prompt_len: usize,
    /// Per-stage `[window + block, kv_dim]` BF16 KV rings.
    pub stage_ring: [DevicePtr; DSPARK_STAGES],
    /// `min(window, next_pos)`.
    pub ring_len: usize,
    /// First sequence position whose ring entry is missing. Equals the
    /// propose's `position` (the first UNPROCESSED position) once the ring is
    /// current — see `ring.rs`.
    pub next_pos: usize,
    /// How many band rows hold the newest committed positions, oldest first:
    /// `rows_through(last_mtp_hidden_idx)`, set by the MODEL
    /// (`TransformerModel::dspark_note_committed_rows`) right before every
    /// propose, because the model is what knows whether the last target step
    /// was a serial decode (1 row) or a verify (`num_accepted + 1` rows).
    /// `after_verify` deliberately does not touch it: it never sees the serial
    /// decodes, and two writers would be two sources.
    pub committed_rows: usize,
    /// Positions that were owed but had no capture row left to rebuild them.
    /// Diagnostic: low acceptance on the Spark is most likely THIS, not the
    /// equations.
    pub gap_positions: usize,
}

impl ProposerState for DsparkProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Device scratch the head OWNS, so nothing it writes depends on an arena
/// buffer's capacity (DFlash sets the same precedent for its tail scratch).
pub(crate) struct DsparkScratch {
    /// `[ATTN_BLOCK_ROWS, nq * hd_mla]` BF16 — Q, padded to the kernel's
    /// 16-row query block.
    pub q: DevicePtr,
    /// `[ATTN_BLOCK_ROWS, nq * hd_mla]` BF16 — attention output, same padding.
    pub attn_out: DevicePtr,
    pub q_latent: DevicePtr,
    /// `[window + block, kv_dim]` BF16 — the KV staging buffer, wide enough
    /// for a whole prefill drain as well as one block.
    pub kv: DevicePtr,
    pub q_rope: DevicePtr,
    pub kv_rope: DevicePtr,
    pub positions: DevicePtr,
    pub win_lo: DevicePtr,
    pub win_hi: DevicePtr,
    pub o_latent: DevicePtr,
    pub o_out: DevicePtr,
    /// `[window + block, hidden]` BF16 — `main_norm(main_proj(capture))`.
    pub main_x: DevicePtr,
    /// `[block, hidden]` BF16 — the collapsed (pre-`norm`) head output.
    pub x: DevicePtr,
    pub normed: DevicePtr,
    /// `[block, vocab]` BF16.
    pub logits: DevicePtr,
    pub markov_embed: DevicePtr,
    pub markov_bias: DevicePtr,
    pub conf_x: DevicePtr,
    pub conf_m: DevicePtr,
    /// `[block + 1]` u32 — the chain, slot 0 the anchor.
    pub draft_tokens_dev: DevicePtr,
    /// `[block]` u32 — the block's input ids, uploaded into `ctx.token_ids`.
    pub token_ids: DevicePtr,
}

/// DeepSeek-V4 native DSpark draft proposer.
pub struct DeepseekV4DsparkHead {
    pub(crate) weights: DeepseekV4DsparkWeights,
    pub(crate) embed_tokens: DenseWeight,
    pub(crate) lm_head: DenseWeight,
    pub(crate) scratch: DsparkScratch,

    /// `config.sliding_window` — the ring depth. PCND: no literal.
    pub(crate) window: usize,
    /// `config.dspark_block_size` (γ).
    pub(crate) block: usize,
    /// `config.dspark_noise_token_id` — fills draft slots 1..γ.
    pub(crate) noise_token: u32,
    /// `config.vocab_size`. 🔴 `mtp_vocab_size` is IGNORED: the Markov tables
    /// are `[vocab_size, rank]` and a narrowed head would mis-index them.
    pub(crate) vocab_size: usize,
    pub(crate) hidden: usize,
    pub(crate) markov_rank: usize,
    pub(crate) n_targets: usize,
    pub(crate) kv_dim: usize,
    /// True when `draft_conf_tau > 0`, i.e. the confidence gate is armed and
    /// the two extra GEMVs per row are worth paying for.
    pub(crate) conf_enabled: bool,
    /// True under EP: the stages are sharded, so the propose needs the
    /// communicator and BOTH ranks must run it.
    pub(crate) ep_comm: bool,

    /// Per-row confidence of the last propose (f32), and the min over the
    /// drafts actually RETURNED, which is what `last_confidence` reports.
    pub(crate) last_conf: Mutex<Vec<f32>>,
    pub(crate) last_conf_bits: AtomicU32,

    // ── Kernel handles, resolved by name (never reached for through the
    //    stage bodies' `pub(super)` handles). ──
    /// HF-vanilla RMSNorm (`out = x * w / rms`) — V4 loads its norm weights
    /// exactly, so the offset-from-1 kernel would compute `x * (1 + w)`.
    pub(crate) rms_norm_vanilla_k: KernelHandle,
    /// Offset-from-1 RMSNorm, used ONLY for the unweighted per-head q_b norm
    /// against the zero-filled `norm_unit_w` (`1 + 0 = 1`).
    pub(crate) rms_norm_offset_k: KernelHandle,
    pub(crate) dense_gemm_k: KernelHandle,
    pub(crate) dense_gemv_k: KernelHandle,
    pub(crate) residual_add_k: KernelHandle,
    pub(crate) argmax_k: KernelHandle,
    pub(crate) batched_embed_k: KernelHandle,
    pub(crate) hc_expand_k: KernelHandle,
    pub(crate) hc_pre_k: KernelHandle,
    pub(crate) hc_post_k: KernelHandle,
    pub(crate) hc_head_k: KernelHandle,
    pub(crate) mla_q_rope_extract_k: KernelHandle,
    pub(crate) mla_q_rope_writeback_k: KernelHandle,
    pub(crate) rope_interleaved_k: KernelHandle,
    pub(crate) rope_interleaved_inv_k: KernelHandle,
    pub(crate) prefill_attn_vis_k: KernelHandle,
    pub(crate) w8a16_gemm_pipelined_k: KernelHandle,
    pub(crate) w8a16_gemv_batch16_strided_k: KernelHandle,
    /// The target capture (`dspark_capture.cu`). Handed to the MODEL, which
    /// owns the capture call sites; kept here because this head is the only
    /// thing that knows the kernel is needed.
    pub(crate) hc_mean_capture_k: KernelHandle,
}

impl DeepseekV4DsparkHead {
    /// Build the proposer from the loaded module plus the SHARED embedding and
    /// LM head.
    ///
    /// Every dimension comes from `config` (PCND). A zero `sliding_window`,
    /// `dspark_block_size` or `vocab_size` is a hard error, not a default.
    pub fn new(
        weights: DeepseekV4DsparkWeights,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &avarok_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let window = config.sliding_window as usize;
        let block = config.dspark_block_size;
        ensure!(
            window > 0,
            "DSpark needs a non-zero `sliding_window`: it IS the ring depth"
        );
        ensure!(
            block > 0 && block <= MAX_BLOCK,
            "DSpark block size {block} outside 1..={MAX_BLOCK}"
        );
        ensure!(config.vocab_size > 0, "DSpark needs a non-zero vocab_size");
        ensure!(
            config.dspark_markov_rank > 0,
            "DSpark needs a non-zero `dspark_markov_rank`: the bigram tables are [vocab, rank]"
        );
        ensure!(
            weights.hc_head.lowrank.is_none(),
            "DSpark reached a low-rank mHC head; this module has only the Sinkhorn dispatch"
        );

        let hidden = config.hidden_size;
        let nq = config.num_attention_heads;
        let n_targets = config.dspark_target_layer_ids.len();
        let (hd_mla, q_lora, o_lora, rope_dim, kv_lora) = {
            let mla = weights.stages[0]
                .body
                .mla
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("DSpark stage 0 has no MLA weights"))?;
            (
                mla.nope + mla.rope,
                mla.q_lora_rank,
                mla.o_lora_rank,
                mla.rope,
                mla.kv_lora_rank,
            )
        };
        // The ring stores ONE latent per position, and the attention reads it
        // as a `head_dim = nope + rope` key. V4-Flash satisfies this by
        // construction (`kv_lora_rank == qk_nope_head_dim + qk_rope_head_dim`);
        // say so rather than let a mismatched checkpoint read past the slot.
        ensure!(
            kv_lora == hd_mla,
            "DSpark ring assumes kv_lora_rank ({kv_lora}) == nope + rope ({hd_mla})"
        );
        for (s, stage) in weights.stages.iter().enumerate() {
            ensure!(
                stage.body.hc.is_some(),
                "DSpark stage {s} has no mHC weights"
            );
            ensure!(
                stage.body.mla.is_some(),
                "DSpark stage {s} has no MLA weights"
            );
        }

        let o_groups = config.o_groups.max(1);
        let rows = window + block;
        let bf16 = 2usize;
        let scratch = DsparkScratch {
            q: gpu.alloc(attn::ATTN_BLOCK_ROWS * nq * hd_mla * bf16)?,
            attn_out: gpu.alloc(attn::ATTN_BLOCK_ROWS * nq * hd_mla * bf16)?,
            q_latent: gpu.alloc(block * q_lora * bf16)?,
            kv: gpu.alloc(rows * hd_mla * bf16)?,
            q_rope: gpu.alloc(attn::ATTN_BLOCK_ROWS * nq * rope_dim * bf16)?,
            kv_rope: gpu.alloc(rows * rope_dim * bf16)?,
            positions: gpu.alloc(rows * 4)?,
            win_lo: gpu.alloc(attn::ATTN_BLOCK_ROWS * 4)?,
            win_hi: gpu.alloc(attn::ATTN_BLOCK_ROWS * 4)?,
            o_latent: gpu.alloc(block * o_groups * o_lora * bf16)?,
            o_out: gpu.alloc(block * hidden * bf16)?,
            main_x: gpu.alloc(rows * hidden * bf16)?,
            x: gpu.alloc(block * hidden * bf16)?,
            normed: gpu.alloc(block * hidden * bf16)?,
            logits: gpu.alloc(block * config.vocab_size * bf16)?,
            markov_embed: gpu.alloc(block * config.dspark_markov_rank * bf16)?,
            markov_bias: gpu.alloc(config.vocab_size * bf16)?,
            conf_x: gpu.alloc(block * bf16)?,
            conf_m: gpu.alloc(block * bf16)?,
            draft_tokens_dev: gpu.alloc((block + 1) * 4)?,
            token_ids: gpu.alloc(block * 4)?,
        };

        Ok(Self {
            weights,
            embed_tokens,
            lm_head,
            scratch,
            window,
            block,
            noise_token: config.dspark_noise_token_id,
            vocab_size: config.vocab_size,
            hidden,
            markov_rank: config.dspark_markov_rank,
            n_targets,
            kv_dim: hd_mla,
            conf_enabled: crate::speculative::draft_conf_tau() > 0.0,
            ep_comm: config.ep_world_size > 1,
            last_conf: Mutex::new(vec![1.0; block]),
            last_conf_bits: AtomicU32::new(1.0f32.to_bits()),
            rms_norm_vanilla_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            rms_norm_offset_k: gpu.kernel("norm", "rms_norm")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            batched_embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_pre_k: gpu.kernel("hyper_connection", "hc_pre")?,
            hc_post_k: gpu.kernel("hyper_connection", "hc_post")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            mla_q_rope_extract_k: gpu.kernel("mla_absorbed", "mla_q_rope_extract_batched")?,
            mla_q_rope_writeback_k: gpu.kernel("mla_absorbed", "mla_q_rope_writeback_batched")?,
            rope_interleaved_k: gpu.kernel("rope", "rope_forward_yarn_interleaved")?,
            rope_interleaved_inv_k: gpu.kernel("rope", "rope_forward_yarn_interleaved_inv")?,
            // Hard lookups, all three: the drafter has no fallback for any of
            // them, so a reduced kernel bundle must fail the boot rather than
            // draft from uninitialised memory.
            prefill_attn_vis_k: gpu
                .kernel("prefill_attn_compressed_vis", "prefill_attn_compressed_vis")?,
            w8a16_gemm_pipelined_k: crate::layers::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w8a16_gemv_batch16_strided_k: crate::layers::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch16_strided",
            ),
            hc_mean_capture_k: gpu.kernel("dspark_capture", "dspark_hc_mean_capture")?,
        })
    }

    /// The capture kernel the MODEL's capture call sites launch.
    pub fn capture_kernel(&self) -> KernelHandle {
        self.hc_mean_capture_k
    }

    pub(crate) fn stage_mla(&self, stage: usize) -> Result<&MlaWeights> {
        self.weights.stages[stage]
            .body
            .mla
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSpark stage {stage} has no MLA weights"))
    }

    pub(crate) fn stage_hc(&self, stage: usize) -> Result<&HcWeights> {
        self.weights.stages[stage]
            .body
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSpark stage {stage} has no mHC weights"))
    }

    pub(crate) fn record_confidence(&self, conf: &[f32]) {
        let mut slot = self.last_conf.lock();
        slot.clear();
        slot.extend_from_slice(conf);
    }
}

/// Read element `i` of a little-endian BF16 byte buffer as f32.
pub(crate) fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    let hi = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
    f32::from_bits((hi as u32) << 16)
}
