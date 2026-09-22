// SPDX-License-Identifier: AGPL-3.0-only

//! One DSpark stage's attention: γ mutually-visible queries over a ring of
//! MAIN-MODEL KV plus the block's own KV.
//!
//! A faithful transcription of `prefill/cache_skip_v4.rs` at `n = γ`, with
//! three differences, all from `DSparkAttention.forward` (`inference/model.py`,
//! l. 800-841):
//!
//! 1. **Keys are the ring, not a paged cache.** The ring holds `main_kv =
//!    rope(kv_norm(wkv(main_x)))` — projections of the TARGET's captured
//!    hiddens, one per committed sequence position, written at `pos % window`.
//!    Nothing here touches `PagedKvCache`.
//! 2. **The block is bidirectional.** `get_dspark_topk_idxs` returns ONE index
//!    list `.expand(bsz, block_size, -1)`, so all γ queries see the same keys:
//!    the valid ring prefix AND all γ block positions, including later ones.
//! 3. **No compressor.** `compress_ratios[n_layers..]` is 0, so the stage runs
//!    the main θ=10000 RoPE table (`mla.main_inv_freq`, mscale 1.0) with no
//!    compressed arm.
//!
//! ## Kernel argument contract (`prefill_attn_compressed_vis`)
//!
//! That kernel's single `seq_len` bounds BOTH the query rows (`q_row <
//! seq_len`) and the key clamp (`if (k_hi >= seq_len) k_hi = seq_len - 1`), and
//! it computes `comp_vis = (q_row + 1) / ratio` unconditionally. So:
//!
//! | argument | value | why |
//! |---|---|---|
//! | `seq_len` | `ring_len + γ` | it is the KEY count; the query count is the grid |
//! | `n_comp` | `0` | no compressed arm |
//! | `ratio` | `1` | **never 0** — `(q_row+1)/ratio` is an unsigned divide |
//! | grid | `(nq, 1, 1)` | one 16-row query block; γ ≤ 16 real rows at 0..γ-1 |
//! | `Q`/`O` | ≥ `BR = 16` rows | the block covers rows 0..15 regardless of γ |
//! | `win_lo/hi[0..γ]` | `(0, ring_len+γ-1)` | bidirectional over the whole visible set |
//! | `win_lo/hi[γ..16]` | `(1, 0)` | empty key loop for the padding rows |

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

use super::DeepseekV4DsparkHead;

/// Query rows one `prefill_attn_compressed_vis` block covers (`BR` in the
/// `.cu`). γ ≤ this, and the remainder are padded with an empty key range.
pub(super) const ATTN_BLOCK_ROWS: usize = 16;

impl DeepseekV4DsparkHead {
    /// `normed` `[γ, hidden]` → this stage's attention output `[γ, hidden]`.
    ///
    /// `position` is the first UNPROCESSED sequence position (the scheduler's
    /// `seq.seq_len`, where `last_token` sits); the γ draft queries sit at
    /// `position .. position + γ`, which is the reference's `start_pos + 1 …
    /// start_pos + γ` since its `start_pos` is the last PROCESSED position
    /// (`ring.rs` module doc). `ring` is THIS stage's per-sequence KV ring (it
    /// lives on `DsparkProposerState`, not on the head).
    pub(super) fn stage_attention(
        &self,
        stage: usize,
        normed: DevicePtr,
        position: usize,
        ring_len: usize,
        ring: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let mla = self.stage_mla(stage)?;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = self.block as u32;
        let h = self.hidden as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let hd_mla = (mla.nope + mla.rope) as u32;
        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let s = &self.scratch;

        // ── 1. Q: wq_a → q_a_norm → wq_b → UNWEIGHTED per-head q_b_norm ──
        self.projection(
            ctx,
            mla.wq_a_fp8.as_ref(),
            &mla.wq_a,
            normed,
            s.q_latent,
            n,
            q_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            s.q_latent,
            &mla.q_a_norm,
            s.q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        self.projection(
            ctx,
            mla.wq_b_fp8.as_ref(),
            &mla.wq_b,
            s.q_latent,
            s.q,
            n,
            nq * hd_mla,
            q_lora,
            stream,
        )?;
        // `q *= rsqrt(mean(q^2) + eps)` per head — the reference's unweighted
        // q_b norm. `rms_norm_k` is the offset-from-1 kernel and `norm_unit_w`
        // is zero-filled, so `1 + 0 = 1`.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_offset_k,
            s.q,
            &DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            s.q,
            n * nq,
            hd_mla,
            eps,
            stream,
        )?;

        // ── 2. KV: wkv → kv_norm (K == V, rope in the tail) ──
        self.projection(
            ctx,
            mla.wkv_a_fp8.as_ref(),
            &mla.wkv_a,
            normed,
            s.kv,
            n,
            kv_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            s.kv,
            &mla.kv_a_norm,
            s.kv,
            n,
            kv_lora,
            eps,
            stream,
        )?;

        // ── 3. Interleaved RoPE at positions `position .. position + γ` ──
        let positions: Vec<u8> = super::ring::block_positions(position, self.block)
            .flat_map(|p| (p as u32).to_le_bytes())
            .collect();
        ctx.gpu.copy_h2d_async(&positions, s.positions, stream)?;
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_k,
            s.q,
            s.q_rope,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_k,
            s.kv,
            s.kv_rope,
            n,
            1,
            hd_mla,
            nope,
            rope,
            hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_interleaved_k,
            s.q_rope,
            s.kv_rope,
            s.positions,
            n,
            nq,
            1,
            rope,
            rope,
            // No compressor on a DSpark stage ⇒ the reference's "main" rope:
            // plain θ=rope_theta, mscale 1.0, no YaRN.
            mla.main_inv_freq,
            1.0f32,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_k,
            s.q_rope,
            s.q,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_k,
            s.kv_rope,
            s.kv,
            n,
            1,
            hd_mla,
            nope,
            rope,
            hd_mla,
            stream,
        )?;

        // ── 4. Block KV into the ring, contiguous with the valid prefix ──
        let base = super::ring::block_base(ring_len);
        ctx.gpu.copy_d2d_async(
            s.kv,
            ring.offset(base * hd_mla as usize * 2),
            self.block * hd_mla as usize * 2,
            stream,
        )?;

        // ── 5. Attention over [valid ring ‖ block KV], bidirectional ──
        let (lo, hi) = super::ring::key_bounds(ring_len, self.block);
        let mut bounds_lo = Vec::with_capacity(ATTN_BLOCK_ROWS * 4);
        let mut bounds_hi = Vec::with_capacity(ATTN_BLOCK_ROWS * 4);
        for row in 0..ATTN_BLOCK_ROWS {
            let (l, hgh) = if row < self.block {
                (lo, hi)
            } else {
                (1u32, 0u32)
            };
            bounds_lo.extend_from_slice(&l.to_le_bytes());
            bounds_hi.extend_from_slice(&hgh.to_le_bytes());
        }
        ctx.gpu.copy_h2d_async(&bounds_lo, s.win_lo, stream)?;
        ctx.gpu.copy_h2d_async(&bounds_hi, s.win_hi, stream)?;

        ensure!(
            self.prefill_attn_vis_k.0 != 0,
            "DSpark attention needs `prefill_attn_compressed_vis`, which did not resolve; there \
             is no causal fallback for a bidirectional draft block"
        );
        let seq_len = (ring_len + self.block) as u32;
        KernelLaunch::new(ctx.gpu, self.prefill_attn_vis_k)
            .grid([nq, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(s.q)
            .arg_ptr(ring)
            // MLA: V == K (the rope tail is part of the key).
            .arg_ptr(ring)
            // `n_comp = 0` ⇒ the compressed loop is empty and these are never
            // dereferenced; they are still a valid pointer, not NULL.
            .arg_ptr(ring)
            .arg_ptr(ring)
            .arg_ptr(mla.attn_sink)
            .arg_ptr(s.attn_out)
            .arg_u32(seq_len)
            .arg_u32(nq)
            .arg_u32(1)
            .arg_u32(hd_mla)
            .arg_u32(0)
            // 🔴 1, never 0: the kernel divides by `ratio` unconditionally.
            .arg_u32(1)
            .arg_ptr(s.win_lo)
            .arg_ptr(s.win_hi)
            .arg_f32(1.0f32 / (hd_mla as f32).sqrt())
            .launch(stream)?;

        // ── 6. De-rotate the output (DeepSeek-V4 eq. 26) ──
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_k,
            s.attn_out,
            s.q_rope,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_interleaved_inv_k,
            s.q_rope,
            s.q_rope,
            s.positions,
            n,
            nq,
            0,
            rope,
            rope,
            mla.main_inv_freq,
            1.0f32,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_k,
            s.q_rope,
            s.attn_out,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;

        // ── 7. Grouped low-rank O projection (block-diagonal wo_a → wo_b) ──
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        if !self.grouped_projection(
            ctx,
            mla.wo_a_fp8.as_ref(),
            s.attn_out,
            s.o_latent,
            n,
            o_groups,
            o_lora,
            group_in,
            stream,
        )? {
            for t in 0..n {
                for g in 0..o_groups {
                    let in_g = s
                        .attn_out
                        .offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                    let w_g = DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                    };
                    let out_g = s
                        .o_latent
                        .offset(((t * latent_dim) + g * o_lora) as usize * 2);
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        in_g,
                        &w_g,
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                }
            }
        }
        self.projection(
            ctx,
            mla.wo_b_fp8.as_ref(),
            &mla.wo_b,
            s.o_latent,
            s.o_out,
            n,
            h,
            latent_dim,
            stream,
        )?;
        Ok(s.o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::ATTN_BLOCK_ROWS;

    /// γ must fit ONE `prefill_attn_compressed_vis` query block, because the
    /// launch uses `grid.y = 1` to keep the padding rows out of the picture.
    /// `MAX_BLOCK` is the head's own ceiling; `BR` is the kernel's.
    #[test]
    fn the_largest_block_fits_one_query_block() {
        const { assert!(super::super::MAX_BLOCK <= ATTN_BLOCK_ROWS) };
        // …and the checkpoint's trained γ is inside that.
        const { assert!(5 <= super::super::MAX_BLOCK) };
    }
}
