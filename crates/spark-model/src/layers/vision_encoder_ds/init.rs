// SPDX-License-Identifier: AGPL-3.0-only

//! [`DeepseekVisionEncoder::new`] — geometry validation, the RoPE
//! inverse-frequency table and every kernel handle.

use anyhow::{Result, ensure};
use avarok_core::config::{VisionConfig, VisionFlavor};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepseekVisionEncoder, DsViTBlock};

impl DeepseekVisionEncoder {
    /// Bind the tower. Every dimension comes from `v` (PCND); nothing here has
    /// a default, and a checkpoint that disagrees with the kernels this target
    /// ships fails at load rather than at the first image.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patch_w: DevicePtr,
        patch_b: DevicePtr,
        blocks: Vec<DsViTBlock>,
        norm_w: DevicePtr,
        aligner_w1_w: DevicePtr,
        aligner_w1_b: DevicePtr,
        aligner_w2_w: DevicePtr,
        aligner_w2_b: DevicePtr,
        v: &VisionConfig,
        hidden_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            v.flavor == VisionFlavor::DeepseekV4,
            "DeepseekVisionEncoder built from a {:?} vision config",
            v.flavor
        );
        let (dim, heads) = (v.hidden_size, v.num_heads);
        ensure!(dim > 0 && heads > 0, "vision dim/heads must be positive");
        ensure!(
            dim.is_multiple_of(heads),
            "vision dim {dim} is not divisible by {heads} heads"
        );
        let head_dim = dim / heads;
        // rotate-half over the head needs an even head_dim; the 2D RoPE then
        // splits that half again into (height, width) angle blocks.
        ensure!(
            head_dim.is_multiple_of(2) && (head_dim / 2).is_multiple_of(2),
            "vision head_dim {head_dim} must be a multiple of 4 for 2D rotate-half RoPE"
        );
        // `rms_norm_vanilla` reads the row as packed BF16 pairs.
        ensure!(dim.is_multiple_of(2), "vision dim {dim} must be even");
        ensure!(v.intermediate_size > 0, "vision_inter_dim must be positive");
        ensure!(v.patch_size > 0, "vision_patch_size must be positive");
        // `ds_vision_unshuffle3` hard-codes the 3×3 tap, exactly as the
        // reference's `Aligner` is only ever built at ratio 3.
        ensure!(
            v.spatial_merge_size == 3,
            "vision_downsample_ratio {} is unsupported: ds_vision_unshuffle3 implements the \
             3x3 pixel unshuffle only",
            v.spatial_merge_size
        );
        ensure!(
            v.out_hidden_size == hidden_size,
            "aligner output {} != model hidden_size {hidden_size}",
            v.out_hidden_size
        );
        ensure!(v.max_n_token > 0, "vision_max_n_token must be positive");
        ensure!(v.rope_theta > 0.0, "vision_rope_theta must be positive");
        ensure!(
            blocks.len() == v.depth,
            "{} ViT blocks bound but vision_n_layers = {}",
            blocks.len(),
            v.depth
        );

        let ds = v.spatial_merge_size;
        // Patch capacity. NOT `derive_max_patches`/`VisionConfig::max_pixels`:
        // `max_pixels` is the Qwen AREA bound that `serve_load.rs` writes into
        // every `config.vision`, and it has no meaning for this tower. The
        // DeepSeek bound is the token budget the resize solver targets, so the
        // widest grid the preprocessor can emit is `ds²·max_n_token` patches.
        let p_max = ds * ds * v.max_n_token;

        // 2D RoPE inverse frequencies — byte-identical to the reference's
        // `get_vision_cos_sin(n_h, n_w, dim = head_dim/2, theta)`:
        // `inv_freq = 1 / theta^(arange(0, rope_dim, 2) / rope_dim)`, i.e.
        // `rope_dim/2` entries. Pinned numerically against
        // `vision_deepseek_ref::rope_cos_sin` in `tests.rs`.
        let rope_dim = head_dim / 2;
        let rope_half = rope_dim / 2;
        let theta = v.rope_theta;
        let rope_inv_freq: Vec<f32> = (0..rope_half)
            .map(|k| (1.0 / theta.powf(2.0 * k as f64 / rope_dim as f64)) as f32)
            .collect();

        Ok(Self {
            patch_w,
            patch_b,
            blocks,
            norm_w,
            aligner_w1_w,
            aligner_w1_b,
            aligner_w2_w,
            aligner_w2_b,
            k_rms: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            k_gemm_tc: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            k_gemm_f32out: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            k_add_bias: gpu.kernel("vision_encoder", "vision_add_bias")?,
            k_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            k_copy: gpu.kernel("vision_encoder", "vision_bf16_copy")?,
            k_f32_bf16: gpu.kernel("vision_encoder", "vision_f32_to_bf16")?,
            k_rope_deint: gpu.kernel("vision_encoder", "vit_rope_deinterleave")?,
            k_softmax: gpu.kernel("vision_encoder", "vit_softmax_rows")?,
            k_scatter: gpu.kernel("vision_encoder", "vit_scatter_head")?,
            k_unshuffle3: gpu.kernel("vision_deepseek", "ds_vision_unshuffle3")?,
            k_gelu_erf: gpu.kernel("vision_deepseek", "ds_gelu_erf_inplace")?,
            k_swiglu: gpu.kernel("vision_deepseek", "ds_swiglu_split")?,
            dim,
            heads,
            head_dim,
            inter: v.intermediate_size,
            patch: v.patch_size,
            ds,
            out_dim: v.out_hidden_size,
            max_n_token: v.max_n_token,
            p_max,
            rope_inv_freq,
            scratch: std::sync::OnceLock::new(),
        })
    }

    /// One patch's flattened pixel count, `3 · patch²`.
    pub fn patch_in(&self) -> usize {
        3 * self.patch * self.patch
    }
}
