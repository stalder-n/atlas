// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek image-block embedding splice.
//!
//! Where the Qwen splice walks a run of identical pad tokens and copies
//! encoder rows in order, a DeepSeek block is a STRUCTURE: a compress pad, a
//! START, a row-pair-interleaved body of IMAGE and NEWLINE slots, an
//! alignment pad and an END. Four of those five types take a learned vector;
//! the IMAGE slots take aligner rows in the block's OWN permutation.
//!
//! ★ SSOT: the layout is never recomputed here.
//! [`block::sentinel_type`] classifies each token and
//! [`block::perm_only`] supplies the permutation — the same functions
//! `block::build_image_block` is built from, which is what the server used to
//! expand the prompt in the first place.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::model::types::TransformerModel;
use crate::vision_deepseek::block;

/// Where one chunk position's hidden row comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DsOverlaySrc {
    /// A learned sentinel vector, by [`block`] type.
    Param(u8),
    /// An absolute row of the vision stage.
    Row(usize),
}

/// Which chunk positions get overwritten, and with what.
///
/// The walk starts at token 0, not at `chunk_start`: the image index, the
/// IMAGE-slot counter and the running row base are all sequential state, so a
/// chunk that begins mid-prompt still has to replay the prefix to know where
/// it is. Only positions `>= chunk_start` are emitted, as offsets INTO THE
/// CHUNK.
pub(super) fn plan_overlay(
    tokens: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    vocab_size: u32,
    grids: &[(usize, usize)],
) -> Result<Vec<(usize, DsOverlaySrc)>> {
    let end = chunk_start + chunk_len;
    ensure!(
        end <= tokens.len(),
        "ds overlay: chunk [{chunk_start}, {end}) is outside a {}-token prompt",
        tokens.len()
    );
    let mut out = Vec::new();
    let mut img_idx = 0usize;
    let mut row_base = 0usize;
    let mut slot_j = 0usize;
    // `Some` exactly while inside an image block.
    let mut span: Option<((usize, usize), Vec<u32>)> = None;

    for (k, &tok) in tokens[..end].iter().enumerate() {
        let Some(ty) = block::sentinel_type(vocab_size, tok) else {
            // Ordinary text. A well-formed block is contiguous, so this also
            // closes nothing: an unterminated span is caught after the loop.
            continue;
        };
        if span.is_none() {
            let Some(&(h, w)) = grids.get(img_idx) else {
                bail!(
                    "ds overlay: the prompt carries more image spans than the {} staged \
                     image(s)",
                    grids.len()
                );
            };
            span = Some(((h, w), block::perm_only(h, w)));
            slot_j = 0;
        }
        let ((h, w), perm) = span.as_ref().expect("span opened above");
        let src = if ty == block::IMAGE {
            let Some(&row) = perm.get(slot_j) else {
                bail!(
                    "ds overlay: image {img_idx} has more IMAGE slots than its {}x{} grid \
                     supplies ({} rows)",
                    h,
                    w,
                    perm.len()
                );
            };
            slot_j += 1;
            DsOverlaySrc::Row(row_base + row as usize)
        } else {
            DsOverlaySrc::Param(ty)
        };
        if k >= chunk_start {
            out.push((k - chunk_start, src));
        }
        if ty == block::IMAGE_END {
            ensure!(
                slot_j == perm.len(),
                "ds overlay: image {img_idx} ended after {slot_j} of {} IMAGE slots",
                perm.len()
            );
            row_base += h * w;
            img_idx += 1;
            span = None;
        }
    }
    ensure!(
        span.is_none(),
        "ds overlay: image {img_idx} has no IMAGE_END within the prefilled range — an image \
         span must be prefilled in a single chunk"
    );
    // Only decidable once the walk covered the whole prompt.
    if end == tokens.len() {
        ensure!(
            img_idx == grids.len(),
            "ds overlay: {img_idx} image span(s) in the prompt but {} staged",
            grids.len()
        );
    }
    Ok(out)
}

/// Overwrite this chunk's sentinel positions in `hidden_dst`.
///
/// Runs where the Qwen splice runs — AFTER `scale_embeddings` — so the
/// spliced rows are never scaled.
pub(super) fn ds_overlay(
    model: &TransformerModel,
    tokens: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    hidden_dst: DevicePtr,
    stream: u64,
) -> Result<()> {
    if !model.ds_vision_active() {
        return Ok(());
    }
    let vocab = model.config.vocab_size as u32;
    let chunk = &tokens[chunk_start..chunk_start + chunk_len];
    if !crate::vision_deepseek::visibility::has_image_span(chunk, vocab) {
        return Ok(());
    }
    let hidden = model.config.hidden_size;
    let params = model.ds_image_params.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "ds overlay: this chunk carries image sentinels but no image_* vectors were \
             bound — the checkpoint declares a vision tower its weights do not ship"
        )
    })?;
    let guard = model.ds_vision_stage.lock();
    let stage = guard.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "ds overlay: this chunk carries image sentinels but nothing is staged. A \
             preempted image request is re-prefilled from its token history, which no \
             longer contains the pixels."
        )
    })?;

    for (pos, src) in plan_overlay(tokens, chunk_start, chunk_len, vocab, &stage.grids)? {
        let from = match src {
            DsOverlaySrc::Param(ty) => params
                .for_type(ty)
                .ok_or_else(|| anyhow::anyhow!("ds overlay: unknown sentinel type {ty}"))?,
            DsOverlaySrc::Row(r) => {
                ensure!(
                    r < stage.n_rows,
                    "ds overlay: aligner row {r} past the {} staged",
                    stage.n_rows
                );
                stage.rows.offset(r * hidden * 2)
            }
        };
        model.gpu.copy_d2d_async(
            from,
            hidden_dst.offset(pos * hidden * 2),
            hidden * 2,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "embed_chunk_ds_tests.rs"]
mod tests;
