// SPDX-License-Identifier: AGPL-3.0-only

//! The staged DeepSeek aligner rows and their EP=2 transport.
//!
//! Rank 0 owns the ViT; rank 1 never loads it. Both ranks run the embedding
//! splice, so the encoded rows have to cross the wire before `prefill_chunk`.
//!
//! # Protocol
//!
//! Immediately after every `0xFFFFFFF0` prefill command's `ep_broadcast_tokens`:
//!
//! 1. `ep_broadcast_u32(n_images)`
//! 2. `ep_broadcast_tokens(&header[1..])` — `(n_llm_h, n_llm_w) × n_images`
//!    then `n_rows`, so `2·n + 1` words whose length BOTH ranks already know
//!    from step 1
//! 3. `comm.broadcast(stage.rows, n_rows · hidden · 2, root = 0)` — straight
//!    out of / into the stage buffer, NEVER through `buffers.scratch()`
//!    (chunk-sized; a 12 MB payload would bail)
//!
//! ★ **Symmetry is the whole game.** Both ranks decide whether this exchange
//! happens from the SAME predicate over the SAME token array —
//! `ds_vision_active() && has_image_span(tokens, vocab_size)`, rank 0 over the
//! array it broadcasts and rank 1 over the array it received. A site that
//! broadcasts a prefill command without calling
//! `Model::ep_broadcast_vision_stage` desyncs the NCCL stream and hangs rank 1.
//!
//! `n_images == 0` is a legal payload meaning "nothing staged". It keeps the
//! ranks in lockstep on the preempt-resume path (which re-prefills a prompt
//! whose images were never re-encoded) instead of hanging; both ranks then
//! fail identically in the overlay with the same span/grid mismatch.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::vision_deepseek::visibility::has_image_span;

/// The encoded aligner rows for the images of ONE prefill, packed in image
/// order.
pub(crate) struct DsVisionStage {
    /// `[cap_rows, hidden]` BF16; the first `n_rows` are live.
    pub rows: DevicePtr,
    /// Rows actually staged, `Σ n_llm_h · n_llm_w`.
    pub n_rows: usize,
    /// Rows the allocation holds. Grown on demand; there is no derivable cap
    /// in the model (`max_prefill_tokens` lives in the server), so inventing a
    /// literal here would be a PCND violation.
    pub cap_rows: usize,
    /// `(n_llm_h, n_llm_w)` per image, in image order.
    pub grids: Vec<(usize, usize)>,
}

/// `[n_images, h0, w0, …, n_rows]`.
pub(crate) fn encode_vision_header(grids: &[(usize, usize)], n_rows: usize) -> Vec<u32> {
    let mut w = Vec::with_capacity(2 * grids.len() + 2);
    w.push(grids.len() as u32);
    for &(h, gw) in grids {
        w.push(h as u32);
        w.push(gw as u32);
    }
    w.push(n_rows as u32);
    w
}

/// Inverse of [`encode_vision_header`], validating everything the receiver
/// would otherwise trust: the length, the grids, and that `n_rows` is exactly
/// what those grids imply.
pub(crate) fn decode_vision_header(words: &[u32]) -> Result<(Vec<(usize, usize)>, usize)> {
    let Some(&n) = words.first() else {
        bail!("ds vision header: empty payload");
    };
    let n = n as usize;
    ensure!(
        words.len() == 2 * n + 2,
        "ds vision header: {n} images needs {} words, got {}",
        2 * n + 2,
        words.len()
    );
    let mut grids = Vec::with_capacity(n);
    let mut total = 0usize;
    for i in 0..n {
        let (h, w) = (words[1 + 2 * i] as usize, words[2 + 2 * i] as usize);
        ensure!(
            h > 0 && w > 0,
            "ds vision header: image {i} has an empty grid {h}x{w}"
        );
        total += h * w;
        grids.push((h, w));
    }
    let n_rows = words[2 * n + 1] as usize;
    ensure!(
        n_rows == total,
        "ds vision header: {n_rows} rows declared but the grids imply {total}"
    );
    Ok((grids, n_rows))
}

/// Row offset of each image inside the stage — the prefix sums of `h · w`.
pub(crate) fn stage_row_bases(grids: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::with_capacity(grids.len());
    let mut acc = 0usize;
    for &(h, w) in grids {
        out.push(acc);
        acc += h * w;
    }
    out
}

impl TransformerModel {
    /// Is this model the DeepSeek-V4 vision tower's host?
    ///
    /// Config-driven, so it is true on BOTH ranks — rank 1 has no encoder but
    /// still receives rows and runs the splice.
    pub(crate) fn ds_vision_active(&self) -> bool {
        self.config
            .vision
            .as_ref()
            .is_some_and(|v| v.flavor == avarok_core::config::VisionFlavor::DeepseekV4)
    }

    /// Does this token array carry a DeepSeek image span? THE predicate both
    /// ranks evaluate to decide whether the vision exchange happens.
    pub(crate) fn ds_tokens_have_image(&self, tokens: &[u32]) -> bool {
        self.ds_vision_active() && has_image_span(tokens, self.config.vocab_size as u32)
    }

    /// The stage buffer, grown in place to hold at least `need_rows`.
    ///
    /// Growth is symmetric across ranks because `n_rows` is known to both
    /// before the bulk broadcast.
    pub(crate) fn ds_stage_ensure(&self, need_rows: usize) -> Result<DevicePtr> {
        let want = need_rows.max(1);
        let mut guard = self.ds_vision_stage.lock();
        if let Some(st) = guard.as_ref()
            && st.cap_rows >= want
        {
            return Ok(st.rows);
        }
        // Take the old entry out BEFORE freeing, so a failed alloc leaves
        // `None` rather than a dangling pointer behind the lock.
        let old = guard.take();
        if let Some(st) = old {
            self.gpu.free(st.rows)?;
        }
        let rows = self.gpu.alloc(want * self.config.hidden_size * 2)?;
        *guard = Some(DsVisionStage {
            rows,
            n_rows: 0,
            cap_rows: want,
            grids: Vec::new(),
        });
        Ok(rows)
    }

    /// Record the freshly encoded rows. Called by the staging path once every
    /// image of the request has been written into the buffer.
    pub(crate) fn ds_stage_publish(&self, n_rows: usize, grids: Vec<(usize, usize)>) -> Result<()> {
        let mut guard = self.ds_vision_stage.lock();
        let st = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ds vision: publish before ds_stage_ensure"))?;
        st.n_rows = n_rows;
        st.grids = grids;
        Ok(())
    }

    /// Rank 0: ship the staged rows for `tokens`. No-op off EP, off DeepSeek
    /// vision, or when `tokens` carries no image span.
    pub(crate) fn ep_send_vision_stage(&self, tokens: &[u32]) -> Result<()> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(());
        };
        if comm.rank() != 0 || !self.ds_tokens_have_image(tokens) {
            return Ok(());
        }
        let (rows, n_rows, header) = {
            let guard = self.ds_vision_stage.lock();
            match guard.as_ref() {
                // A sentinel-bearing prompt with nothing staged (the preempt
                // re-prefill path). Announce "no images" so rank 1 takes the
                // same branch instead of blocking on a broadcast that never
                // comes; both then fail in the overlay with one message.
                None => (DevicePtr::NULL, 0usize, encode_vision_header(&[], 0)),
                Some(st) => (
                    st.rows,
                    st.n_rows,
                    encode_vision_header(&st.grids, st.n_rows),
                ),
            }
        };
        self.ep_broadcast_u32(header[0])?;
        self.ep_broadcast_tokens(&header[1..])?;
        if n_rows > 0 {
            // The encode ran on the default stream; NCCL is not ordered
            // against it, so without this the collective can read rows the
            // ViT has not finished writing.
            self.gpu.synchronize(self.gpu.default_stream())?;
            comm.broadcast(rows.0, n_rows * self.config.hidden_size * 2, 0)?;
        }
        Ok(())
    }

    /// Worker ranks: receive what [`Self::ep_send_vision_stage`] sent.
    pub(crate) fn ep_recv_vision_stage(&self) -> Result<()> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(());
        };
        let n_images = self.ep_broadcast_u32(0)? as usize;
        let rest = self.ep_broadcast_tokens(&vec![0u32; 2 * n_images + 1])?;
        let mut words = Vec::with_capacity(rest.len() + 1);
        words.push(n_images as u32);
        words.extend_from_slice(&rest);
        let (grids, n_rows) = decode_vision_header(&words)?;
        if n_rows == 0 {
            *self.ds_vision_stage.lock() = None;
            return Ok(());
        }
        let rows = self.ds_stage_ensure(n_rows)?;
        comm.broadcast(rows.0, n_rows * self.config.hidden_size * 2, 0)?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.ds_stage_publish(n_rows, grids)
    }
}

#[cfg(test)]
#[path = "ep_vision_tests.rs"]
mod tests;
