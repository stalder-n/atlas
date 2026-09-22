// SPDX-License-Identifier: AGPL-3.0-only

//! `encode_image` — one image, patches → aligner rows — plus the GEMM/norm/
//! copy primitives the block loop and the aligner share.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{DeepseekVisionEncoder, DsViTBlock};
use crate::vision_deepseek::geometry::n_llm;

/// The ViT's RMSNorm epsilon.
///
/// The reference's `vision.py::RMSNorm` defaults to `1e-6` and the ViT never
/// overrides it. It is deliberately NOT `ModelConfig::rms_norm_eps`, which
/// carries the LLM's own (unclamped) value.
const DS_VIT_RMS_EPS: f32 = 1e-6;

impl DeepseekVisionEncoder {
    /// `C[m,n] = A[m,k] @ B[n,k]ᵀ + bias[n]`, BF16 in and out: a tensor-core
    /// GEMM followed by a row-broadcast bias add.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ds_gemm_bias(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: DevicePtr,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.ds_gemm_nobias(gpu, a, b, c, m, n, k, stream)?;
        KernelLaunch::new(gpu, self.k_add_bias)
            .grid([div_ceil(m * n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(bias)
            .arg_u32(m)
            .arg_u32(n)
            .launch(stream)
    }

    /// `C[m,n] = A[m,k] @ B[n,k]ᵀ`. The ViT MLP's `w1`/`w2` carry no bias.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ds_gemm_nobias(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_gemm_tc)
            .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }

    /// Element copy over `n` BF16 values.
    fn ds_copy(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        n: u32,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n)
            .launch(s)
    }

    /// Vanilla RMSNorm (`out = w · x / rms`), out-of-place capable, here
    /// always in place on `x`.
    fn ds_rms(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        rows: u32,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rms)
            .grid([rows, 1, 1])
            .block([(self.dim as u32).min(1024), 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(x)
            .arg_u32(self.dim as u32)
            .arg_f32(DS_VIT_RMS_EPS)
            .launch(s)
    }

    /// One ViT block, in place on `buf_h1`.
    ///
    /// `x = x + attn(norm1(x)); x = x + mlp(norm2(x))` — the reference's
    /// `Block.forward`, with `buf_h2` holding the residual and `buf_wide` /
    /// `buf_act` the wide intermediates.
    fn ds_vit_block(
        &self,
        blk: &DsViTBlock,
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let (p32, dim32) = (p as u32, self.dim as u32);
        let n_h = (p * self.dim) as u32;

        // ── attention ──
        self.ds_copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?;
        self.ds_rms(gpu, s.buf_h1, blk.norm1_w, p32, stream)?;
        self.ds_gemm_bias(
            gpu,
            s.buf_h1,
            blk.qkv_w,
            blk.qkv_b,
            s.buf_wide,
            p32,
            3 * dim32,
            dim32,
            stream,
        )?;
        self.ds_attention(gpu, s.buf_wide, s.buf_h1, p32, stream)?;
        self.ds_gemm_bias(
            gpu, s.buf_h1, blk.wo_w, blk.wo_b, s.buf_wide, p32, dim32, dim32, stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_wide)
            .arg_ptr(s.buf_h2)
            .arg_u32(n_h)
            .launch(stream)?;
        self.ds_copy(gpu, s.buf_wide, s.buf_h1, n_h, stream)?;

        // ── MLP ──
        self.ds_copy(gpu, s.buf_h1, s.buf_h2, n_h, stream)?;
        self.ds_rms(gpu, s.buf_h1, blk.norm2_w, p32, stream)?;
        let inter32 = self.inter as u32;
        self.ds_gemm_nobias(
            gpu,
            s.buf_h1,
            blk.w1_w,
            s.buf_wide,
            p32,
            2 * inter32,
            dim32,
            stream,
        )?;
        // gate = first `inter` columns, up = last `inter` (torch `chunk(2)`).
        // Writes a DISJOINT buffer: `buf_wide` is still being read.
        KernelLaunch::new(gpu, self.k_swiglu)
            .grid([div_ceil(p32 * inter32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_wide)
            .arg_ptr(s.buf_act)
            .arg_u32(p32)
            .arg_u32(inter32)
            .launch(stream)?;
        self.ds_gemm_nobias(
            gpu, s.buf_act, blk.w2_w, s.buf_h1, p32, dim32, inter32, stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_h1)
            .arg_ptr(s.buf_h2)
            .arg_u32(n_h)
            .launch(stream)
    }

    /// Encode ONE image: `[n_vit_h·n_vit_w, 3·patch²]` f32 patches →
    /// `[n_llm_h·n_llm_w, out_dim]` BF16 aligner rows at `dst`.
    ///
    /// Returns the LLM grid the rows are laid out on. `image_index` only names
    /// the `AVAROK_DS_VISION_DUMP` files, so the Spark compare script can pair
    /// them with the request's images.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_image(
        &self,
        pixels: &[f32],
        n_vit_h: usize,
        n_vit_w: usize,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
        image_index: usize,
    ) -> Result<(usize, usize)> {
        self.scratch_init(gpu)?;
        let p = n_vit_h * n_vit_w;
        ensure!(p > 0, "ds vision: empty patch grid {n_vit_h}x{n_vit_w}");
        // A `Result`, not a `debug_assert`: a release build compiles the assert
        // out and the oversized H2D copy becomes CUDA_ERROR_INVALID_VALUE from
        // deep inside the scheduler, naming neither vision nor a size.
        ensure!(
            p <= self.p_max,
            "ds vision: image is {n_vit_h}x{n_vit_w} patches ({p} total) but this encoder \
             holds {} (= downsample_ratio^2 * vision_max_n_token). The preprocessor's \
             resize solver targets that budget, so a grid this large means the two \
             disagree.",
            self.p_max
        );
        ensure!(
            pixels.len() == p * self.patch_in(),
            "ds vision: {} pixel values for {p} patches of {} each",
            pixels.len(),
            self.patch_in()
        );
        let (n_llm_h, n_llm_w) = (n_llm(n_vit_h, self.ds), n_llm(n_vit_w, self.ds));
        let n_tok = n_llm_h * n_llm_w;
        ensure!(
            n_tok <= self.max_n_token,
            "ds vision: {n_llm_h}x{n_llm_w} = {n_tok} LLM tokens exceeds vision_max_n_token {}",
            self.max_n_token
        );

        let s = self.scratch();
        self.build_rope_cossin_into(
            n_vit_h,
            n_vit_w,
            s.buf_rope_cos,
            s.buf_rope_sin,
            gpu,
            stream,
        )?;

        // SAFETY: `pixels` is a live `&[f32]` and the byte length is exactly
        // `pixels.len() * size_of::<f32>()`, so the view cannot leave the
        // slice. `f32` has no invalid bit patterns and `u8` has alignment 1;
        // the view is read-only and dies before `pixels`.
        let px_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        gpu.copy_h2d_async(px_bytes, s.buf_f32, stream)?;
        let n_px = (p * self.patch_in()) as u32;
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_px, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_f32)
            .arg_ptr(s.buf_patch_bf16)
            .arg_u32(n_px)
            .launch(stream)?;

        // patch embed → buf_h1
        self.ds_gemm_bias(
            gpu,
            s.buf_patch_bf16,
            self.patch_w,
            self.patch_b,
            s.buf_h1,
            p as u32,
            self.dim as u32,
            self.patch_in() as u32,
            stream,
        )?;

        for blk in &self.blocks {
            self.ds_vit_block(blk, p, gpu, stream)?;
        }
        self.ds_rms(gpu, s.buf_h1, self.norm_w, p as u32, stream)?;

        self.ds_aligner(gpu, n_vit_h, n_vit_w, n_llm_h, n_llm_w, dst, stream)?;
        self.maybe_dump(
            gpu,
            dst,
            (n_vit_h, n_vit_w),
            (n_llm_h, n_llm_w),
            image_index,
            stream,
        )?;
        Ok((n_llm_h, n_llm_w))
    }

    /// `AVAROK_DS_VISION_DUMP=<dir>`: snapshot this image's aligner rows as raw
    /// BF16 plus a one-line JSON sidecar, for the Spark bring-up compare.
    fn maybe_dump(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        vit: (usize, usize),
        llm: (usize, usize),
        idx: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(dir) = std::env::var("AVAROK_DS_VISION_DUMP")
            .ok()
            .filter(|d| !d.is_empty())
        else {
            return Ok(());
        };
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; llm.0 * llm.1 * self.out_dim * 2];
        gpu.copy_d2h(dst, &mut buf)?;
        std::fs::create_dir_all(&dir)?;
        let base = std::path::Path::new(&dir);
        std::fs::write(base.join(format!("image_{idx}.bf16")), &buf)?;
        let meta = format!(
            "{{\"n_vit_h\":{},\"n_vit_w\":{},\"n_llm_h\":{},\"n_llm_w\":{}}}",
            vit.0, vit.1, llm.0, llm.1
        );
        std::fs::write(base.join(format!("image_{idx}.json")), meta)?;
        Ok(())
    }
}
