// SPDX-License-Identifier: AGPL-3.0-only

//! The aligner: 3×3 pixel unshuffle → `w1` → exact-erf GELU → `w2`.
//!
//! Port of the reference's `vision.py::Aligner.forward`, whose
//! `F.pad(..., (0, -n_w % r, 0, -n_h % r))` + `F.unfold(r, stride=r)` is
//! exactly what `ds_vision_unshuffle3` does in one pass: zero-pad the grid up
//! to a multiple of 3 and gather each 3×3 tap into the CHANNEL-slow feature
//! layout `c·9 + kh·3 + kw`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::DeepseekVisionEncoder;

impl DeepseekVisionEncoder {
    /// `buf_h1[n_vit_h·n_vit_w, dim]` → `dst[n_llm_h·n_llm_w, out_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ds_aligner(
        &self,
        gpu: &dyn GpuBackend,
        n_vit_h: usize,
        n_vit_w: usize,
        n_llm_h: usize,
        n_llm_w: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let n_tok = (n_llm_h * n_llm_w) as u32;
        let unshuffled = (self.dim * self.ds * self.ds) as u32;
        let out = self.out_dim as u32;

        // One block per OUTPUT token, a strided thread loop over its dim·9
        // features — the geometry `vision_spatial_merge` uses next door.
        KernelLaunch::new(gpu, self.k_unshuffle3)
            .grid([n_tok, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_h1)
            .arg_ptr(s.buf_unshuffle)
            .arg_u32(n_vit_h as u32)
            .arg_u32(n_vit_w as u32)
            .arg_u32(n_llm_h as u32)
            .arg_u32(n_llm_w as u32)
            .arg_u32(self.dim as u32)
            .launch(stream)?;

        self.ds_gemm_bias(
            gpu,
            s.buf_unshuffle,
            self.aligner_w1_w,
            self.aligner_w1_b,
            s.buf_align_hidden,
            n_tok,
            out,
            unshuffled,
            stream,
        )?;

        // Exact erf GELU — `nn.Linear` + bare `F.gelu` in the reference, whose
        // default `approximate="none"` is the erf form, NOT the tanh
        // approximation `vision_gelu` implements for Qwen.
        KernelLaunch::new(gpu, self.k_gelu_erf)
            .grid([div_ceil(n_tok * out, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_align_hidden)
            .arg_u32(n_tok * out)
            .launch(stream)?;

        self.ds_gemm_bias(
            gpu,
            s.buf_align_hidden,
            self.aligner_w2_w,
            self.aligner_w2_b,
            dst,
            n_tok,
            out,
            out,
            stream,
        )
    }
}
