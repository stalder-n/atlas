// SPDX-License-Identifier: AGPL-3.0-only

//! Lazy ViT scratch allocation and the host-built 2D RoPE tables.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepseekVisionEncoder, DsVisionScratch, f32_to_bf16_bits};

impl DeepseekVisionEncoder {
    /// Allocate the scratch group. Sizes depend only on the tower geometry, so
    /// the group is built once and reused by every image after.
    fn build_scratch(&self, gpu: &dyn GpuBackend) -> Result<DsVisionScratch> {
        let p = self.p_max;
        let (dim, inter, hd) = (self.dim, self.inter, self.head_dim);
        let t = self.max_n_token;
        // `buf_wide` carries BOTH the QKV projection (3·dim) and the fused
        // gate‖up (2·inter); size it to whichever is wider.
        let wide = (3 * dim).max(2 * inter);
        Ok(DsVisionScratch {
            buf_f32: gpu.alloc(p * self.patch_in() * 4)?,
            buf_patch_bf16: gpu.alloc(p * self.patch_in() * 2)?,
            buf_h1: gpu.alloc(p * dim * 2)?,
            buf_h2: gpu.alloc(p * dim * 2)?,
            buf_wide: gpu.alloc(p * wide * 2)?,
            buf_act: gpu.alloc(p * inter * 2)?,
            buf_rope_cos: gpu.alloc(p * hd * 2)?,
            buf_rope_sin: gpu.alloc(p * hd * 2)?,
            buf_qr: gpu.alloc(p * self.heads * hd * 2)?,
            buf_kr: gpu.alloc(p * self.heads * hd * 2)?,
            buf_vt: gpu.alloc(p * self.heads * hd * 2)?,
            // O(p²), and the single largest buffer here: reused across the
            // head loop, which is why the heads must stay on one stream.
            buf_scores: gpu.alloc(p * p * 4)?,
            buf_probs: gpu.alloc(p * p * 2)?,
            buf_o_stage: gpu.alloc(p * hd * 2)?,
            buf_unshuffle: gpu.alloc(t * dim * self.ds * self.ds * 2)?,
            buf_align_hidden: gpu.alloc(t * self.out_dim * 2)?,
        })
    }

    /// The scratch group, allocated on the FIRST image and never at load.
    ///
    /// `OnceLock` rather than a flag: the encode path takes `&self`, and two
    /// racing images must converge on one group.
    pub(crate) fn scratch_init(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.scratch.get().is_none() {
            let s = self.build_scratch(gpu)?;
            // A racing caller may have won; theirs is equally valid.
            let _ = self.scratch.set(s);
            tracing::info!(
                "DeepSeek vision scratch allocated on first image: {} patches",
                self.p_max
            );
        }
        Ok(())
    }

    /// Scratch accessor for the encode path. Panics when the entry point did
    /// not call [`Self::scratch_init`] first — a wiring bug, not a runtime
    /// condition.
    pub(crate) fn scratch(&self) -> &DsVisionScratch {
        self.scratch
            .get()
            .expect("ds vision scratch: encode entry must call scratch_init(gpu) first")
    }

    /// Build the per-patch 2D rotary cos/sin tables for an `n_h × n_w` grid
    /// and upload them as BF16 `[n_h·n_w, head_dim]`.
    ///
    /// Row-major positions — `h` outer, `w` inner — matching the reference's
    /// `get_vision_cos_sin`, which builds `hpos/wpos` with
    /// `torch.arange(n_h).unsqueeze(1)` / `torch.arange(n_w).unsqueeze(0)` and
    /// reshapes to `(-1, 2, 1)`. This is NOT a merge-window order.
    ///
    /// Layout per row, with `inv_n = head_dim/4` frequencies:
    ///   `[cos(h·f)×inv_n ‖ cos(w·f)×inv_n ‖ cos(h·f)×inv_n ‖ cos(w·f)×inv_n]`
    ///
    /// The duplicated second half is what `vit_rope_deinterleave`'s rotate-half
    /// consumes: for lane `d ≥ head_dim/2` it reads `cos[d]`, and the reference's
    /// `apply_rotary` pairs `head[j]` with `head[rope_dim + j]` against the SAME
    /// angle, so `cos[d] == cos[d - head_dim/2]` must hold.
    pub(crate) fn build_rope_cossin_into(
        &self,
        n_h: usize,
        n_w: usize,
        cos_dst: DevicePtr,
        sin_dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = n_h * n_w;
        let hd = self.head_dim;
        let half = hd / 2;
        let inv_n = self.rope_inv_freq.len();
        debug_assert_eq!(inv_n * 2, half);

        let mut cos_bf16 = vec![0u16; p * hd];
        let mut sin_bf16 = vec![0u16; p * hd];
        for gh in 0..n_h {
            for gw in 0..n_w {
                let off = (gh * n_w + gw) * hd;
                let (row, col) = (gh as f32, gw as f32);
                for k in 0..inv_n {
                    let (rs, rc) = (row * self.rope_inv_freq[k]).sin_cos();
                    let (cs, cc) = (col * self.rope_inv_freq[k]).sin_cos();
                    for base in [0, half] {
                        cos_bf16[off + base + k] = f32_to_bf16_bits(rc);
                        sin_bf16[off + base + k] = f32_to_bf16_bits(rs);
                        cos_bf16[off + base + inv_n + k] = f32_to_bf16_bits(cc);
                        sin_bf16[off + base + inv_n + k] = f32_to_bf16_bits(cs);
                    }
                }
            }
        }
        // SAFETY (both): each Vec is a live `vec![0u16; p * hd]` and each byte
        // length is derived from that same Vec's `len()`, so neither view can
        // leave its allocation. The `vec!` zeroed every element, `u16` has no
        // invalid bit patterns, `u8` has alignment 1, and both views are
        // read-only and die before their Vecs.
        let cos_b: &[u8] = unsafe {
            std::slice::from_raw_parts(cos_bf16.as_ptr() as *const u8, cos_bf16.len() * 2)
        };
        let sin_b: &[u8] = unsafe {
            std::slice::from_raw_parts(sin_bf16.as_ptr() as *const u8, sin_bf16.len() * 2)
        };
        gpu.copy_h2d_async(cos_b, cos_dst, stream)?;
        gpu.copy_h2d_async(sin_b, sin_dst, stream)
    }
}
