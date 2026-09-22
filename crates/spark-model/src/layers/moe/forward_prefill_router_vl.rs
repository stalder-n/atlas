// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash-Vision-Exp prefill routing: the `_vl` twins of the two
//! DeepSeek-V4 router dispatches in `forward_prefill.rs`.
//!
//! The Vision-Exp checkpoint ships a SECOND expert selection bias,
//! `ffn.gate.bias_vl`, used for image tokens only — a token whose id is a
//! sentinel (`>= vocab_size`; sentinels travel in the token stream above the
//! vocabulary and can never be sampled, `lm_head` is `[vocab_size]`). Both
//! `_vl` kernels are byte-identical to their text twin for a text token.
//!
//! Text checkpoints (0731 and the whole support matrix) load no `bias_vl`, so
//! `correction_bias_vl_dev` is `None`, no `moe_router_vl` lookup is ever
//! issued, and the dispatch below is the one that exists today.

use anyhow::{Result, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Resolve the two `moe_router_vl` kernels — ONLY for a layer that actually
/// carries a `bias_vl`.
///
/// `MoeLayer::new_with_hash` is the constructor for EVERY MoE model, and
/// `moe_router_vl` is registered only under
/// `kernels/gb10/deepseek-v4-flash/nvfp4/` with no `[expected_absent]`
/// declaration anywhere (absence must fail the boot audit). An unconditional
/// probe would therefore add a REQUIRED failed-lookup row to Qwen3.x /
/// Gemma-4 / MiniMax / Nemotron / Mistral boots. Gating on the model's own
/// config is the remedy `layers::try_kernel`'s doc prescribes, and it lets the
/// Vision-Exp lookup be hard — correct for a module with no fallback path.
pub(super) fn vl_router_kernels(
    gpu: &dyn GpuBackend,
    enabled: bool,
) -> Result<(KernelHandle, KernelHandle)> {
    if !enabled {
        return Ok((KernelHandle(0), KernelHandle(0)));
    }
    Ok((
        gpu.kernel("moe_router_vl", "moe_topk_sqrtsoftplus_batched_vl")?,
        gpu.kernel("moe_router_vl", "moe_hash_route_batched_vl")?,
    ))
}

/// Which of the four prefill routers a DeepSeek-V4 MoE layer dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillRouter {
    /// `moe_hash_route::moe_hash_route_batched` — text checkpoint, hash layer.
    Hash,
    /// `moe_router_vl::moe_hash_route_batched_vl` — Vision-Exp, hash layer.
    HashVl,
    /// `moe_topk_sqrt::moe_topk_sqrtsoftplus_batched` — text, non-hash layer.
    SqrtSoftplus,
    /// `moe_router_vl::moe_topk_sqrtsoftplus_batched_vl` — Vision-Exp,
    /// non-hash layer.
    SqrtSoftplusVl,
}

/// The dispatch decision table: hash-ness picks the family, `bias_vl` AND
/// device token ids together pick the `_vl` twin.
///
/// The ids are the ONLY way to tell an image sentinel (`id >= vocab_size`)
/// from a vocabulary token, so without them the `_vl` kernel has nothing to
/// branch on and the text kernel is the honest choice.
///
/// INVARIANT THIS RELIES ON (owned by the prefill-embedding unit, not here):
/// any chunk whose tokens include a sentinel also populates
/// `ForwardContext.token_ids`. Sentinels only enter the stream through the
/// server-side image-block expansion, which is the same path that uploads the
/// device token ids. If that ever stops holding, a sentinel-bearing chunk with
/// `token_ids == None` would route images with the TEXT bias — a silent
/// mis-route — and this `false` arm must become a hard error instead.
pub(super) fn select_prefill_router(
    has_tid2eid: bool,
    has_bias_vl: bool,
    has_token_ids: bool,
) -> PrefillRouter {
    match (has_tid2eid, has_bias_vl && has_token_ids) {
        (true, true) => PrefillRouter::HashVl,
        (true, false) => PrefillRouter::Hash,
        (false, true) => PrefillRouter::SqrtSoftplusVl,
        (false, false) => PrefillRouter::SqrtSoftplus,
    }
}

impl MoeLayer {
    /// Hash-routing dispatch for batched prefill (DeepSeek-V4 hash_moe layer):
    /// static `tid2eid[token_id]` selection, sqrtsoftplus-weighted, with the
    /// learned `bias_vl` arm for image sentinels on a Vision-Exp checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_hash_prefill(
        &self,
        gate_logits: DevicePtr,
        tid2eid: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let token_ids = ctx.token_ids.ok_or_else(|| {
            anyhow!(
                "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill grouped)"
            )
        })?;
        let router = select_prefill_router(true, self.correction_bias_vl_dev.is_some(), true);
        if let (PrefillRouter::HashVl, Some(bias_vl)) = (router, self.correction_bias_vl_dev) {
            return ops::moe_hash_route_batched_vl(
                ctx.gpu,
                self.moe_hash_route_batched_vl_k,
                gate_logits,
                tid2eid,
                token_ids,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                bias_vl,
                ctx.config.vocab_size as u32,
                n,
                stream,
            );
        }
        ops::moe_hash_route_batched(
            ctx.gpu,
            self.moe_hash_route_batched_k,
            gate_logits,
            tid2eid,
            token_ids,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            n,
            stream,
        )
    }

    /// Sqrt-softplus + correction-bias dispatch for batched prefill
    /// (DeepSeek-V4 non-hash MoE layer). On a Vision-Exp checkpoint carrying
    /// device token ids, the per-token selection bias becomes
    /// `tok >= vocab_size ? bias_vl : bias`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_sqrtsoftplus_prefill(
        &self,
        gate_logits: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let router = select_prefill_router(
            false,
            self.correction_bias_vl_dev.is_some(),
            ctx.token_ids.is_some(),
        );
        if let (PrefillRouter::SqrtSoftplusVl, Some(bias_vl), Some(token_ids)) =
            (router, self.correction_bias_vl_dev, ctx.token_ids)
        {
            return ops::moe_topk_sqrtsoftplus_batched_vl(
                ctx.gpu,
                self.moe_topk_sqrtsoftplus_batched_vl_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                bias_vl,
                token_ids,
                ctx.config.vocab_size as u32,
                n,
                stream,
            );
        }
        ops::moe_topk_sqrtsoftplus_batched(
            ctx.gpu,
            self.moe_topk_sqrtsoftplus_batched_k,
            gate_logits,
            bias,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            n,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "forward_prefill_router_vl_tests.rs"]
mod router_vl_tests;
