// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 projection dispatch for the DSpark stage attention.
//!
//! A local copy of `qwen3_attention/prefill/v4_projections.rs`: that one is
//! `pub(super)` inside `qwen3_attention` and gates on kernel handles the layer
//! resolved for itself, while this head resolves its own by name (the plan's
//! rule — never reach into another module's private handles).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

use super::DeepseekV4DsparkHead;

impl DeepseekV4DsparkHead {
    /// `w8a16_gemm_pipelined` on a native FP8 block-scaled weight, else the
    /// BF16 `dense_gemm`. A local copy of `prefill/v4_projections.rs`'s arm —
    /// that one is `pub(super)` inside `qwen3_attention` and gates on kernel
    /// handles this head resolves for itself.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn projection(
        &self,
        ctx: &ForwardContext,
        fp8: Option<&Fp8Weight>,
        bf16: &DenseWeight,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(w) = fp8.filter(|w| fp8_compatible(w, n, k))
            && ctx.levers.v4_fp8_prefill
            && self.w8a16_gemm_pipelined_k.0 != 0
            && m > 0
        {
            return ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                input,
                w.weight,
                w.row_scale,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            input,
            bf16,
            output,
            m,
            n,
            k,
            stream,
        )
    }

    /// Block-diagonal FP8 `wo_a`, preserving the token-major pitches.
    /// Returns false BEFORE any launch when the native path does not apply.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn grouped_projection(
        &self,
        ctx: &ForwardContext,
        weight: Option<&Fp8Weight>,
        input: DevicePtr,
        output: DevicePtr,
        tokens: u32,
        groups: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = weight.filter(|w| fp8_compatible(w, groups * n, k)) else {
            return Ok(false);
        };
        if !ctx.levers.v4_fp8_prefill
            || self.w8a16_gemv_batch16_strided_k.0 == 0
            || !n.is_multiple_of(128)
            || tokens == 0
            || groups == 0
        {
            return Ok(false);
        }
        for group in 0..groups {
            ops::w8a16_gemv_batch16_strided(
                ctx.gpu,
                self.w8a16_gemv_batch16_strided_k,
                input.offset((group * k) as usize * 2),
                w.weight.offset((group * n * k) as usize),
                w.row_scale
                    .offset((group * (n / 128) * (k / 128)) as usize * 4),
                output.offset((group * n) as usize * 2),
                tokens.min(16),
                n,
                k,
                groups * k,
                groups * n,
                stream,
            )?;
        }
        Ok(true)
    }
}

fn fp8_compatible(weight: &Fp8Weight, n: u32, k: u32) -> bool {
    weight.n == n
        && weight.k == k
        && n > 0
        && k > 0
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && weight.scale_format == WeightQuantFormat::Fp8BlockScaled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_compat_requires_matching_block_scaled_geometry() {
        let mut w = Fp8Weight {
            weight: DevicePtr(128),
            row_scale: DevicePtr(256),
            n: 1024,
            k: 4096,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        assert!(fp8_compatible(&w, 1024, 4096));
        assert!(!fp8_compatible(&w, 512, 4096));
        assert!(!fp8_compatible(&w, 1024, 2048));
        // Not block-scaled: the pipelined GEMM would read the wrong scale
        // layout, so the BF16 arm must take over BEFORE any launch.
        w.scale_format = WeightQuantFormat::Fp8PerRow;
        assert!(!fp8_compatible(&w, 1024, 4096));
        w.scale_format = WeightQuantFormat::Fp8BlockScaled;
        w.n = 1025;
        assert!(!fp8_compatible(&w, 1025, 4096));
    }
}
