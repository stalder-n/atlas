// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 projections retain BF16 activations and FP32 128×128 block scales.
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

fn compatible(weight: &Fp8Weight, n: u32, k: u32) -> bool {
    weight.n == n
        && weight.k == k
        && n > 0
        && k > 0
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && weight.scale_format == WeightQuantFormat::Fp8BlockScaled
}

impl Qwen3AttentionLayer {
    /// Return false only BEFORE launch, so execution errors never trigger a fallback.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4_fp8_projection(
        &self,
        ctx: &ForwardContext,
        weight: Option<&Fp8Weight>,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(weight) = weight.filter(|weight| compatible(weight, n, k)) else {
            return Ok(false);
        };
        if !ctx.levers.v4_fp8_prefill
            // This scalar implementation is shipped only in the GB10 V4 target.
            // It is independent of the optional TC-attention kernel and toggle.
            || self.prefill_attn_compressed_k.0 == 0
            || self.w8a16_gemm_pipelined_k.0 == 0
            || m == 0
        {
            return Ok(false);
        }
        ops::w8a16_gemm_pipelined(
            ctx.gpu,
            self.w8a16_gemm_pipelined_k,
            input,
            weight.weight,
            weight.row_scale,
            output,
            m,
            n,
            k,
            stream,
        )?;
        Ok(true)
    }

    /// Preserve token-major input/output pitches while selecting each block-diagonal group.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4_fp8_grouped_projection(
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
        let Some(weight) = weight.filter(|weight| compatible(weight, groups * n, k)) else {
            return Ok(false);
        };
        if !ctx.levers.v4_fp8_prefill
            // This scalar implementation is shipped only in the GB10 V4 target.
            // It is independent of the optional TC-attention kernel and toggle.
            || self.prefill_attn_compressed_k.0 == 0
            || self.w8a16_gemv_batch16_strided_k.0 == 0
            || !n.is_multiple_of(128)
            || tokens == 0
            || groups == 0
        {
            return Ok(false);
        }
        for start in (0..tokens).step_by(16) {
            for group in 0..groups {
                ops::w8a16_gemv_batch16_strided(
                    ctx.gpu,
                    self.w8a16_gemv_batch16_strided_k,
                    input.offset(((start * groups + group) * k) as usize * 2),
                    weight.weight.offset((group * n * k) as usize),
                    weight
                        .row_scale
                        .offset((group * (n / 128) * (k / 128)) as usize * 4),
                    output.offset(((start * groups + group) * n) as usize * 2),
                    (tokens - start).min(16),
                    n,
                    k,
                    groups * k,
                    groups * n,
                    stream,
                )?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_projection_requires_matching_block_scaled_shape() {
        let mut weight = Fp8Weight {
            weight: DevicePtr(128),
            row_scale: DevicePtr(256),
            n: 256,
            k: 384,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        assert!(compatible(&weight, 256, 384));
        assert!(!compatible(&weight, 128, 384));
        assert!(!compatible(&weight, 256, 128));
        weight.scale_format = WeightQuantFormat::Fp8PerRow;
        assert!(!compatible(&weight, 256, 384));
        weight.scale_format = WeightQuantFormat::Fp8BlockScaled;
        weight.n = 129;
        assert!(!compatible(&weight, 129, 384));
    }
}
