// SPDX-License-Identifier: AGPL-3.0-only

//! One DSpark stage's mHC sites around the bespoke attention and the reused
//! MoE.
//!
//! Reference (`inference/model.py`, `DSparkBlock.forward`, l. 895): for
//! `start_pos > 0` a DSpark stage is the PLAIN `Block.forward` — the same
//! six-step mHC sequence every V4 layer runs — with `DSparkAttention` in place
//! of `MLA`. So this transcribes `qwen3_attention/trait_impl/prefill_inner.rs`
//! `prefill_inner_hc` at `n = γ`, minus everything that is about a target
//! layer (the highway seed, `hc_head`, the Gemma post-norms, the CSA
//! compressor, the paged cache):
//!
//! ```text
//! hc_pre(attn) → attn_norm → DSparkAttention → hc_post
//! hc_pre(ffn)  → ffn_norm  → MoE             → hc_post
//! ```
//!
//! `hc_expand` (the highway seed) happens ONCE for the whole block, in
//! [`super::propose`], and `hc_head` (the collapse) ONCE after stage 2, in
//! [`super::markov_confidence`] — exactly as `forward_spec` splits them
//! across `forward_embed` / the stage loop / `forward_head`.

use anyhow::Result;

use crate::layer::ForwardContext;
use crate::layers::ops;
use spark_runtime::gpu::DevicePtr;

use super::DeepseekV4DsparkHead;

impl DeepseekV4DsparkHead {
    /// Run stage `stage` over the γ block rows held in `ctx.buffers.hc_streams()`.
    ///
    /// `moe_ctx` must be the DERIVED context carrying the block's token ids
    /// (all `< vocab_size`, so U2.3's `_vl` router selects `bias`, not
    /// `bias_vl` — byte-identical to the text path, which is what the
    /// reference does: `image_mask = input_ids >= vocab_size` is all-false for
    /// `[last_token, noise × (γ-1)]`).
    pub(super) fn stage_forward(
        &self,
        stage: usize,
        position: usize,
        ring_len: usize,
        ring: DevicePtr,
        moe_ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let ctx = moe_ctx;
        let n = self.block as u32;
        let h = self.hidden as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc = self.stage_hc(stage)?;
        let body = &self.weights.stages[stage];

        let hc_streams = ctx.buffers.hc_streams();
        let collapsed = ctx.buffers.hidden_states();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();

        // ── attention sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.attn,
            hc,
            collapsed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            collapsed,
            &body.attn_norm,
            normed,
            n,
            h,
            eps,
            stream,
        )?;
        let attn_out =
            self.stage_attention(stage, normed, position, ring_len, ring, ctx, stream)?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            attn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h,
            stream,
        )?;

        // ── FFN sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.ffn,
            hc,
            collapsed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            collapsed,
            &body.ffn_norm,
            normed,
            n,
            h,
            eps,
            stream,
        )?;
        // 🪤 At γ = 5 this falls to `forward_batched` (the grouped-GEMM
        // threshold is 64), which re-streams the expert weights per token.
        // Three stages of that per propose is the throughput risk the bring-up
        // doc frames: 2.99 accepted/forward is necessary, not sufficient.
        body.body
            .ffn_component()
            .forward_prefill(normed, self.block, ctx, stream)
            .map_err(|e| anyhow::anyhow!("DSpark stage {stage} MoE failed: {e}"))?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ctx.buffers.moe_output(),
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h,
            stream,
        )?;
        Ok(())
    }
}
