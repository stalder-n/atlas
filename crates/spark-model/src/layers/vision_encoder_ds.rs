// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash-Vision-Exp vision tower (`vision.*`) + aligner
//! (`aligner.*`).
//!
//! A SEPARATE encoder from [`crate::layers::vision_encoder::VisionEncoder`]
//! on purpose: the two towers share the `VisionConfig` struct and almost
//! nothing else. DeepSeek is RMSNorm(1e-6) + SwiGLU + 2D RoPE with a 3×3
//! pixel-unshuffle aligner; Qwen is LayerNorm + tanh-GELU + learned APE +
//! deepstack with a 2×2 spatial merge. Reference: DeepSeek's
//! `inference/vision.py` at revision `86f746b36186f0e567729a5c06a8c918caba82a9`,
//! ported to f32 host arithmetic in [`crate::vision_deepseek_ref`].
//!
//! # What this produces
//!
//! One image → `[n_llm_h · n_llm_w, hidden_size]` BF16 aligner rows, written
//! into a caller-supplied device pointer. The caller (the model's vision
//! staging path) owns the packing; this type owns only the arithmetic.
//!
//! # Blind code
//!
//! Every CUDA launch here was written without a GPU or `nvcc`. The host tests
//! pin the kernel sequence, the launch geometry, the scalar arguments and the
//! RoPE tables — not one numeric ViT output. First real signal is the Spark
//! bring-up comparing `AVAROK_DS_VISION_DUMP` rows against
//! [`crate::vision_deepseek_ref`].

use spark_runtime::gpu::{DevicePtr, KernelHandle};

use crate::vision_deepseek::block;

mod aligner;
mod attn;
mod forward;
mod init;
mod scratch;

#[cfg(test)]
#[path = "vision_encoder_ds/tests.rs"]
mod tests;

/// One ViT block's weights. No norm biases and no MLP biases: the reference's
/// `RMSNorm` has weight only, and its `MLP` builds both linears with
/// `bias=False`.
pub struct DsViTBlock {
    /// `vision.blocks.{i}.norm1.weight` — `[dim]`.
    pub norm1_w: DevicePtr,
    /// `vision.blocks.{i}.attn.wqkv.weight` — `[3·dim, dim]`.
    pub qkv_w: DevicePtr,
    /// `vision.blocks.{i}.attn.wqkv.bias` — `[3·dim]`.
    pub qkv_b: DevicePtr,
    /// `vision.blocks.{i}.attn.wo.weight` — `[dim, dim]`.
    pub wo_w: DevicePtr,
    /// `vision.blocks.{i}.attn.wo.bias` — `[dim]`.
    pub wo_b: DevicePtr,
    /// `vision.blocks.{i}.norm2.weight` — `[dim]`.
    pub norm2_w: DevicePtr,
    /// `vision.blocks.{i}.mlp.w1.weight` — `[2·inter, dim]` (gate ‖ up).
    pub w1_w: DevicePtr,
    /// `vision.blocks.{i}.mlp.w2.weight` — `[dim, inter]`.
    pub w2_w: DevicePtr,
}

/// The four learned per-sentinel vectors the model splices into non-IMAGE
/// positions of an image block. `[hidden_size]` BF16 each.
///
/// Both EP ranks hold these: the splice runs on every rank, and they are four
/// vectors, not a tower.
pub struct DsImageParams {
    /// `image_start`.
    pub start: DevicePtr,
    /// `image_pad`.
    pub pad: DevicePtr,
    /// `image_newline`.
    pub newline: DevicePtr,
    /// `image_end`.
    pub end: DevicePtr,
}

impl DsImageParams {
    /// The vector for sentinel type `ty`, mirroring the reference's
    /// `params = torch.stack([image_start, image_pad, image_pad,
    /// image_newline, image_end])` — index 2 (`IMAGE`) is `image_pad` there
    /// too, and is always overwritten by an aligner row before it is read.
    pub fn for_type(&self, ty: u8) -> Option<DevicePtr> {
        match ty {
            block::IMAGE_START => Some(self.start),
            block::IMAGE_PAD | block::IMAGE => Some(self.pad),
            block::IMAGE_NEWLINE => Some(self.newline),
            block::IMAGE_END => Some(self.end),
            _ => None,
        }
    }
}

/// Per-image ViT scratch, allocated as one group on the FIRST image.
///
/// ~175 MB at the production geometry (`p_max = 3456`), of which `buf_scores`
/// alone is 47.8 MB — the ViT attention materialises a full `[p, p]` f32 score
/// matrix per head. A text-only serve never allocates a byte of it.
pub struct DsVisionScratch {
    /// `[p_max, 3·patch²]` f32 — the uploaded pixel patches.
    pub buf_f32: DevicePtr,
    /// `[p_max, 3·patch²]` BF16 — the same, converted.
    pub buf_patch_bf16: DevicePtr,
    /// `[p_max, dim]` BF16 — the residual stream.
    pub buf_h1: DevicePtr,
    /// `[p_max, dim]` BF16 — the saved residual.
    pub buf_h2: DevicePtr,
    /// `[p_max, max(3·dim, 2·inter)]` BF16 — QKV and the fused gate‖up.
    pub buf_wide: DevicePtr,
    /// `[p_max, inter]` BF16 — the SwiGLU output. Disjoint from `buf_wide`,
    /// which is its input.
    pub buf_act: DevicePtr,
    /// `[p_max, head_dim]` BF16.
    pub buf_rope_cos: DevicePtr,
    /// `[p_max, head_dim]` BF16.
    pub buf_rope_sin: DevicePtr,
    /// `[heads, p_max, head_dim]` BF16 — rotated Q, head-contiguous.
    pub buf_qr: DevicePtr,
    /// `[heads, p_max, head_dim]` BF16 — rotated K, head-contiguous.
    pub buf_kr: DevicePtr,
    /// `[heads, head_dim, p_max]` BF16 — V, transposed.
    pub buf_vt: DevicePtr,
    /// `[p_max, p_max]` f32 — raw QKᵀ scores for ONE head.
    pub buf_scores: DevicePtr,
    /// `[p_max, p_max]` BF16 — softmaxed probabilities for ONE head.
    pub buf_probs: DevicePtr,
    /// `[p_max, head_dim]` BF16 — one head's attention output before scatter.
    pub buf_o_stage: DevicePtr,
    /// `[max_n_token, dim·ds²]` BF16 — the 3×3 pixel-unshuffle output.
    pub buf_unshuffle: DevicePtr,
    /// `[max_n_token, out_dim]` BF16 — the aligner's hidden activation.
    pub buf_align_hidden: DevicePtr,
}

/// The DeepSeek ViT tower plus its aligner.
pub struct DeepseekVisionEncoder {
    /// `vision.patch_embed.proj.weight` — `[dim, 3·patch²]`.
    patch_w: DevicePtr,
    /// `vision.patch_embed.proj.bias` — `[dim]`.
    patch_b: DevicePtr,
    blocks: Vec<DsViTBlock>,
    /// `vision.norm.weight` — `[dim]`.
    norm_w: DevicePtr,
    /// `aligner.w1.weight` — `[out_dim, dim·ds²]`.
    aligner_w1_w: DevicePtr,
    /// `aligner.w1.bias` — `[out_dim]`.
    aligner_w1_b: DevicePtr,
    /// `aligner.w2.weight` — `[out_dim, out_dim]`.
    aligner_w2_w: DevicePtr,
    /// `aligner.w2.bias` — `[out_dim]`.
    aligner_w2_b: DevicePtr,

    // ── kernel handles (all HARD: `kernels/gb10/deepseek-v4-flash/MODEL.toml`
    // deliberately declares no `[expected_absent]` for the vision path, because
    // none of these has a fallback) ──
    /// `rms_norm_vanilla::rms_norm_vanilla`. NOT `norm::rms_norm`, which is
    /// Qwen3-Next's offset-from-1 form `x · (1 + w) / rms`; DeepSeek's
    /// `vision.py::RMSNorm` is vanilla `w · x / rms`.
    k_rms: KernelHandle,
    /// `gemm::dense_gemm_bf16_pipelined` — tensor-core `A[m,k] @ B[n,k]ᵀ`.
    k_gemm_tc: KernelHandle,
    /// `gemm::dense_gemm_bf16_f32out` — raw QKᵀ scores.
    k_gemm_f32out: KernelHandle,
    /// `vision_encoder::vision_add_bias`.
    k_add_bias: KernelHandle,
    /// `vision_encoder::vision_add_inplace`.
    k_add: KernelHandle,
    /// `vision_encoder::vision_bf16_copy`.
    k_copy: KernelHandle,
    /// `vision_encoder::vision_f32_to_bf16`.
    k_f32_bf16: KernelHandle,
    /// `vision_encoder::vit_rope_deinterleave`.
    k_rope_deint: KernelHandle,
    /// `vision_encoder::vit_softmax_rows`.
    k_softmax: KernelHandle,
    /// `vision_encoder::vit_scatter_head`.
    k_scatter: KernelHandle,
    /// `vision_deepseek::ds_vision_unshuffle3`.
    k_unshuffle3: KernelHandle,
    /// `vision_deepseek::ds_gelu_erf_inplace`.
    k_gelu_erf: KernelHandle,
    /// `vision_deepseek::ds_swiglu_split`.
    k_swiglu: KernelHandle,

    // ── geometry (PCND: every one is read from `VisionConfig`) ──
    /// `vision_dim` (1024).
    pub dim: usize,
    /// `vision_n_heads` (16).
    pub heads: usize,
    /// `dim / heads` (64).
    pub head_dim: usize,
    /// `vision_inter_dim` (2816).
    pub inter: usize,
    /// `vision_patch_size` (14).
    pub patch: usize,
    /// `vision_downsample_ratio` (3).
    pub ds: usize,
    /// `hidden_size` — the aligner's output width (4096).
    pub out_dim: usize,
    /// `vision_max_n_token` (384).
    pub max_n_token: usize,
    /// Patch capacity, `ds² · max_n_token` (3456).
    pub p_max: usize,
    /// `vision_rope_theta^(-2k/rope_dim)` for `k in 0..head_dim/4`.
    rope_inv_freq: Vec<f32>,
    /// Allocated on the FIRST image, never at load.
    scratch: std::sync::OnceLock<DsVisionScratch>,
}

/// Round an f32 to BF16 bits, round-to-nearest-even.
///
/// A local copy rather than a shared helper: the Qwen tower's twin
/// (`vision_encoder::enc_impl::f32_to_bf16_bits`) is `pub(super)` inside a
/// module this unit must not edit.
#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0; // canonical quiet NaN
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}
