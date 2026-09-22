// SPDX-License-Identifier: AGPL-3.0-only

//! The ViT block's bidirectional SDPA over one image.
//!
//! `vit_rope_deinterleave` once, then per head `dense_gemm_bf16_f32out` →
//! `vit_softmax_rows` → `dense_gemm_bf16_pipelined` → `vit_scatter_head`.
//!
//! All of them are the tower-agnostic kernels from `vision_encoder.cu`: they
//! take `H`/`D` as arguments and carry no Qwen geometry. The softmax folds
//! `rsqrt(D)`, which IS the reference's `F.scaled_dot_product_attention`
//! default scale `1/√head_dim`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::DeepseekVisionEncoder;

impl DeepseekVisionEncoder {
    /// `o[seq, H·D] = SDPA(rope(q), rope(k), v)` over one image's QKV block.
    ///
    /// Every launch shares `stream`, so head `h`'s
    /// GEMM1→softmax→GEMM2→scatter completes before head `h+1` reuses
    /// `buf_scores`/`buf_probs`/`buf_o_stage`. Do NOT split heads across
    /// streams without per-head score buffers.
    ///
    /// No `seq <= 1024` bound: the Qwen twin asserts one because ITS buffers
    /// were sized for that; these are `p_max`-sized and `encode_image` has
    /// already refused anything larger.
    pub(super) fn ds_attention(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr,
        o: DevicePtr,
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let h_n = self.heads as u32;
        let d = self.head_dim as u32;
        let hd = self.dim as u32; // H·D, the output row stride

        // (1) rope + deinterleave + V-transpose, all heads at once.
        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(s.buf_qr)
            .arg_ptr(s.buf_kr)
            .arg_ptr(s.buf_vt)
            .arg_ptr(s.buf_rope_cos)
            .arg_ptr(s.buf_rope_sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .launch(stream)?;

        let qk_head = (seq * d) as usize;
        for head in 0..self.heads {
            let qr_h = s.buf_qr.offset(head * qk_head * 2);
            let kr_h = s.buf_kr.offset(head * qk_head * 2);
            let vt_h = s.buf_vt.offset(head * qk_head * 2);
            let o_h = o.offset(head * self.head_dim * 2);

            // (2) raw QKᵀ, f32 out. `dense_gemm_bf16_f32out` is TILE=16.
            KernelLaunch::new(gpu, self.k_gemm_f32out)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h)
                .arg_ptr(kr_h)
                .arg_ptr(s.buf_scores)
                .arg_u32(seq)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // (3) row softmax with `rsqrt(D)` folded in.
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.buf_scores)
                .arg_ptr(s.buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // (4) P·V — `Vt` is pre-transposed, so the A·Bᵀ GEMM yields P·V.
            KernelLaunch::new(gpu, self.k_gemm_tc)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(s.buf_probs)
                .arg_ptr(vt_h)
                .arg_ptr(s.buf_o_stage)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(seq)
                .launch(stream)?;

            // (5) scatter the contiguous head output into its interleaved slot.
            KernelLaunch::new(gpu, self.k_scatter)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(())
    }
}
