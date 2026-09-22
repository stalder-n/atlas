// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 **native DSpark** draft-module loader (`mtp.{0,1,2}.*`).
//!
//! The 0731 and Vision-Exp releases ship a THREE-stage block drafter, not the
//! single next-token MTP module of `nvidia/DeepSeek-V4-Flash-NVFP4`. The two
//! are disjoint checkpoint families and are told apart by one tensor:
//!
//! | family | marker | loader |
//! |---|---|---|
//! | legacy 1-module MTP | `mtp.0.enorm.weight` | [`super::mtp::load_v4_mtp_module`] |
//! | native 3-stage DSpark | `mtp.0.main_proj.weight` | this file |
//!
//! Reference (`inference/model.py`, the authority):
//!
//! ```text
//! forward       (l. 986): h = layer(...); if i in target_layer_ids:
//!                         main_hiddens.append(h.mean(dim=2))   # over hc streams
//! forward_spec  (l.1014): h, main_x = mtp[0].forward_embed(main_hidden, ids)
//!                         for layer in mtp: h = layer(h, start_pos, ids, main_x)
//!                         return mtp[-1].forward_head(h, ids)
//! forward_embed (l. 901): main_x = main_norm(main_proj(main_hidden))
//!                         x = embed([last, noise x (γ-1)]) -> hc_expand
//! forward_head  (l. 909): x = hc_head(h); logits = lm_head(norm(x))
//!                         markov chain over γ rows; confidence from the PRE-norm x
//! ```
//!
//! So the stage-scoped tensors are: `main_proj`/`main_norm` on stage 0,
//! `norm`/`hc_head_*`/`markov_head`/`confidence_head` on stage 2, and a full V4
//! body (MLA + mHC + 256-expert MoE) on each of the three.
//!
//! ## EP posture — the OPPOSITE of the legacy module
//!
//! `DeepseekV4DsparkHead::needs_comm()` is true under EP, so the stages are
//! **EP-sharded and loaded on EVERY rank** (`force_all_experts = false`) and
//! the block's own all-reduce assembles the routed sum — exactly like the text
//! stack, and exactly unlike `load_v4_mtp_module`, which replicates all 256
//! experts on rank 0 and runs with `comm: None`. Getting this backwards costs
//! either ~5 GB/rank of duplicated experts or a half-computed routed sum.

use anyhow::{Context, Result, bail, ensure};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layers::qwen3_attention::{HcHeadWeights, Qwen3AttentionLayer};
use crate::weight_map::{DenseWeight, dense, dense_auto};

/// The reference's `n_mtp_layers` for this family. Not a config knob: the
/// checkpoint ships exactly three and the preflight contract
/// (`preflight::deepseek_v4::check_dspark_quant_contract`) refuses anything
/// else, so a constant here and a `{0,1,2}` there are one fact.
pub const DSPARK_STAGES: usize = 3;

/// One DSpark stage: a full V4 block body plus its two RMSNorm weights.
///
/// The norms are held here rather than read back off `body` because the head
/// drives the sites itself (`hc_pre → attn → hc_post → hc_pre → ffn → hc_post`)
/// and `Qwen3AttentionLayer::input_norm` / `post_attn_norm` are `pub(super)`.
/// `DenseWeight` is `Copy` and these are BF16 (so `dense_auto` hands back the
/// store pointer unchanged): the body and the stage hold the SAME device
/// buffer, not a second copy of it.
pub struct DsparkStage {
    /// MLA + mHC + MoE (`bias` and, on Vision-Exp, `bias_vl`), assembled from
    /// the `mtp.{s}` prefix at `layer_idx = num_hidden_layers + s`.
    pub body: Qwen3AttentionLayer,
    /// `mtp.{s}.attn_norm.weight` — HF-vanilla RMSNorm, loaded exactly.
    pub attn_norm: DenseWeight,
    /// `mtp.{s}.ffn_norm.weight` — HF-vanilla RMSNorm, loaded exactly.
    pub ffn_norm: DenseWeight,
}

/// The loaded native DSpark module. Embedding and `lm_head` are SHARED with
/// the parent model and supplied at proposer-build time, as for every other
/// drafter in the tree.
pub struct DeepseekV4DsparkWeights {
    pub stages: [DsparkStage; DSPARK_STAGES],
    /// `mtp.0.main_proj` `[hidden, hidden * n_targets]`, FP8 block-scaled on
    /// disk and dequanted to BF16 by `dense_auto`. Consumes the concatenated
    /// capture of the `dspark_target_layer_ids` hc-stream means.
    pub main_proj: DenseWeight,
    /// `mtp.0.main_norm.weight` `[hidden]`.
    pub main_norm: DenseWeight,
    /// `mtp.2.norm.weight` `[hidden]` — the final norm before the shared head.
    pub norm: DenseWeight,
    /// `mtp.2.hc_head_{fn,base,scale}` — the module's OWN Sinkhorn head mHC.
    pub hc_head: HcHeadWeights,
    /// `mtp.2.markov_head.markov_w1.weight` `[vocab, rank]` BF16 — the bigram
    /// EMBEDDING table, indexed by the previous chain token.
    pub markov_w1: DenseWeight,
    /// `mtp.2.markov_head.markov_w2.weight` `[vocab, rank]` BF16 — the bigram
    /// HEAD, producing the additive logit bias.
    pub markov_w2: DenseWeight,
    /// `mtp.2.confidence_head.proj.weight` `[1, hidden + rank]` BF16.
    ///
    /// 🔴 NO BIAS. DeepSeek's is `Linear(dim + rank, 1, bias=False)`; DFlash's
    /// confidence head has one, and reusing that code path would read a
    /// tensor this checkpoint does not ship.
    pub confidence_w: DenseWeight,
}

/// Load the native 3-stage DSpark module, or `Ok(None)` when this checkpoint
/// is not one.
///
/// Gate: `dspark_block_size > 0` **and** `mtp.0.main_proj.weight` present.
/// `preflight::deepseek_v4` gates on the same tensor, so preflight and loader
/// cannot disagree about which family a checkpoint belongs to.
pub fn load_v4_dspark_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Option<DeepseekV4DsparkWeights>> {
    if config.dspark_block_size == 0 || !store.contains("mtp.0.main_proj.weight") {
        return Ok(None);
    }
    ensure!(
        config.hc_mult > 0,
        "DeepSeek-V4 DSpark requires the mHC highway (hc_mult > 0); the stage bodies and the \
         head collapse both run through it"
    );
    ensure!(
        config.dspark_target_layer_ids.len() == DSPARK_STAGES,
        "DeepSeek-V4 DSpark expects {DSPARK_STAGES} target layers, config declares {:?}",
        config.dspark_target_layer_ids,
    );

    let mut built = Vec::with_capacity(DSPARK_STAGES);
    for stage in 0..DSPARK_STAGES {
        built.push(
            load_stage(store, config, gpu, layer_kv_dtypes, stage)
                .with_context(|| format!("DeepSeek-V4 DSpark stage {stage}"))?,
        );
    }
    let stages: [DsparkStage; DSPARK_STAGES] = match built.try_into() {
        Ok(s) => s,
        Err(_) => bail!("DeepSeek-V4 DSpark: stage vector was not {DSPARK_STAGES} long"),
    };

    let main_proj = dense_auto(store, "mtp.0.main_proj.weight", gpu)
        .context("DeepSeek-V4 DSpark: mtp.0.main_proj")?;
    let main_norm = dense_auto(store, "mtp.0.main_norm.weight", gpu)?;
    let norm = dense_auto(store, "mtp.2.norm.weight", gpu)?;
    let hc_head = load_stage_hc_head(store, config, gpu)?;
    // The Markov tables and the confidence projection are plain BF16 — loaded
    // EXACTLY, never `dense_auto`'s dequant path, so a future FP8 repack of
    // them fails loudly instead of silently changing the chain's arithmetic.
    let markov_w1 = dense(store, "mtp.2.markov_head.markov_w1.weight")?;
    let markov_w2 = dense(store, "mtp.2.markov_head.markov_w2.weight")?;
    let confidence_w = dense(store, "mtp.2.confidence_head.proj.weight")?;
    ensure!(
        !store.contains("mtp.2.confidence_head.proj.bias"),
        "DeepSeek-V4 DSpark: the confidence head is Linear(.., 1, bias=False); this checkpoint \
         ships a bias, so the reference changed and the head must be re-derived"
    );

    let (local_start, local_end) = config.local_expert_range();
    tracing::info!(
        "DeepSeek-V4 native DSpark module loaded: {DSPARK_STAGES} stages (MLA + mHC + MoE), \
         EP rank {}/{} owns experts [{}..{}) per stage, block_size={}, markov_rank={}, \
         target_layers={:?}",
        config.ep_rank,
        config.ep_world_size,
        local_start,
        local_end,
        config.dspark_block_size,
        config.dspark_markov_rank,
        config.dspark_target_layer_ids,
    );

    Ok(Some(DeepseekV4DsparkWeights {
        stages,
        main_proj,
        main_norm,
        norm,
        hc_head,
        markov_w1,
        markov_w2,
        confidence_w,
    }))
}

/// `mtp.2.hc_head_{fn,base,scale}` — the DSpark module's own head mixer,
/// distinct from the model-level one the target's last layer consumes.
fn load_stage_hc_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcHeadWeights> {
    let hc = config.hc_mult;
    let hc_dim = hc * config.hidden_size;
    Ok(HcHeadWeights {
        hc_fn: super::assemble::load_hc_f32(
            store,
            &["mtp.2.hc_head_fn".to_string()],
            hc * hc_dim,
            gpu,
        )?,
        hc_base: super::assemble::load_hc_f32(store, &["mtp.2.hc_head_base".to_string()], hc, gpu)?,
        hc_scale: super::assemble::load_hc_f32(
            store,
            &["mtp.2.hc_head_scale".to_string()],
            1,
            gpu,
        )?,
        // DeepSeek keeps the Sinkhorn mixer; `None` selects it. The head
        // refuses a low-rank variant at construction (there is no dispatch).
        lowrank: None,
    })
}

/// One stage body, assembled through the same `assemble_layer_concrete` the
/// target layers use.
///
/// `layer_idx = num_hidden_layers + stage` is deliberate: `compress_ratios`,
/// the hash-layer predicate and the per-layer KV dtype all index by it, and an
/// out-of-range index makes all three fall to the safe defaults (no
/// compressor, no hash routing, BF16 KV) — which is exactly the reference's
/// `DSparkBlock` (its `layer_id >= n_hash_layers`, so it is never hash-routed,
/// and `compress_ratios[43..46] == 0`, so `mla.compressor` is `None` and the
/// forward uses the main θ=10000 RoPE table).
fn load_stage(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
    stage: usize,
) -> Result<DsparkStage> {
    let prefix = format!("mtp.{stage}");
    let ap = format!("{prefix}.attn");
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };

    // Mirrors the V4-Flash arm of `load_all_layers` (o_lora_rank > 0): direct
    // KV projection + grouped low-rank O projection, so the V3 absorption
    // tensors (w_uk_t / w_uv / w_qk_absorbed / the block-diagonals) are NULL.
    let attn_norm = dense_auto(store, &format!("{prefix}.attn_norm.weight"), gpu)?;
    let ffn_norm = dense_auto(store, &format!("{prefix}.ffn_norm.weight"), gpu)?;
    let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
    let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
    let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;
    let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
    let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;
    let wo_a = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
    let wo_b = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;

    let mut yarn_inv_freq = DevicePtr::NULL;
    yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

    let body = super::assemble::assemble_layer_concrete(
        config.num_hidden_layers + stage,
        &prefix,
        // 🔴 FALSE, unlike the legacy MTP module: the DSpark stages are
        // EP-SHARDED and run on both ranks with the block's own all-reduce.
        false,
        attn_norm,
        ffn_norm,
        wq_a,
        None, // wq_a_nvfp4 — V4 attention is FP8/BF16, never NVFP4
        wq_b,
        None, // wq_b_nvfp4
        q_a_norm,
        wkv_a,
        None, // wkv_a_nvfp4
        null, // wkv_b — unused for V4-Flash
        kv_a_norm,
        wo_b, // o_dense
        None, // o_nvfp4
        null, // w_uk_t
        null, // w_uv
        null, // wq_b_rope
        null, // w_qk_absorbed
        null, // w_uk_block_diag
        null, // w_uv_block_diag
        yarn_inv_freq,
        wo_a,
        // The stage body must NOT carry a head mixer: the DSpark head collapses
        // the highway itself, once, after stage 2 — `mtp.2.hc_head_*` lives on
        // `DeepseekV4DsparkWeights`, not on any stage.
        None,
        store,
        config,
        gpu,
        layer_kv_dtypes,
    )?;

    Ok(DsparkStage {
        body,
        attn_norm,
        ffn_norm,
    })
}
