// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the two `moe_router_vl` kernels
//! (`kernels/gb10/deepseek-v4-flash/nvfp4/moe_router_vl.cu`) — the VL twins of
//! `common/moe_topk_sqrtsoftplus.cu::moe_topk_sqrtsoftplus_batched` and
//! `common/moe_hash_route.cu::moe_hash_route_batched`.
//!
//! Both twins carry a second, image-only expert selection bias (`bias_vl`) plus
//! the device token ids and `vocab_size` needed to tell an image sentinel
//! (`id >= vocab_size`) from a vocabulary token. Everything else — grid (N,1,1),
//! block (256,1,1), argument order — is the text kernel's, verbatim.
//!
//! ARITY IS PINNED IN THREE PLACES: the `.cu` signature,
//! `avarok-kernels/tests/kernel_arity.rs` and
//! `avarok-kernels/tests/deepseek_v4_kernel_audit.rs` both pin 11 parameters
//! for each kernel, and the two wrappers below push exactly 11 arguments.
//! Change one, change all four — an argument-order slip is invisible on this
//! host (no nvcc, no GPU) and surfaces on the Spark as
//! `CUDA_ERROR_INVALID_VALUE` or silently mis-routed tokens.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Batched sqrt-softplus routing with a per-modality selection bias
/// (DeepSeek-V4-Flash-Vision-Exp prefill, non-hash MoE layers).
///
/// Per token `t`, the selection bias is `token_ids[t] >= vocab_size ? bias_vl :
/// bias`; the weights still come from the pre-bias sqrt-softplus scores, as in
/// the text kernel. Byte-identical to
/// [`super::moe_topk_sqrtsoftplus_batched`] for a text token.
///
/// Kernel: `moe_router_vl::moe_topk_sqrtsoftplus_batched_vl`
/// Grid: (N, 1, 1)  Block: (256, 1, 1) — the block shape is load-bearing
/// (`BLOCK_SIZE` is a compile-time 256 in the cross-warp top-k reduction).
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrtsoftplus_batched_vl(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    bias_vl: DevicePtr,
    token_ids: DevicePtr,
    vocab_size: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .arg_ptr(bias_vl)
        .arg_ptr(token_ids)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// Batched hash routing with a learned VL arm for image tokens
/// (DeepSeek-V4-Flash-Vision-Exp prefill, hash MoE layers).
///
/// Text tokens (`token_ids[t] < vocab_size`) run the verbatim
/// `moe_hash_route_batched` body. Image sentinels have no row in `tid2eid`
/// (it is indexed by vocabulary id), so they fall back to a learned top-k of
/// `sqrtsoftplus(logits) + bias_vl` with the same pre-bias weights,
/// `norm_topk_prob` normalisation and `routed_scaling_factor` scaling.
///
/// Kernel: `moe_router_vl::moe_hash_route_batched_vl`
/// Grid: (N, 1, 1)  Block: (256, 1, 1) — see above; the text arm is
/// thread-0-only but the image arm is whole-block.
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route_batched_vl(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    tid2eid: DevicePtr,
    token_ids: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    bias_vl: DevicePtr,
    vocab_size: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(tid2eid)
        .arg_ptr(token_ids)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .arg_ptr(bias_vl)
        .arg_u32(vocab_size)
        .launch(stream)
}
