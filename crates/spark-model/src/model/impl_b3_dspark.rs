// SPDX-License-Identifier: AGPL-3.0-only

//! Native-DSpark target capture, branched out of the three DFlash capture
//! functions in `impl_b3.rs`.
//!
//! Branching there rather than adding nine `try_dspark_*` twins keeps all nine
//! existing call sites (`decode_a2/a3`, `prefill_a`, `prefill_b/forward_layers`,
//! `verify_b/c/c2/d/e/fused`) untouched. Every branch is gated on
//! `self.dspark_capture.is_some()`, which is `None` for every model that is not
//! running the DSpark proposer — so DFlash's capture bytes are unchanged.
//!
//! Two things differ from the DFlash capture it sits beside:
//!
//! 1. **The source is the mHC highway**, `buffers.hc_streams()`
//!    (`[T, hc_mult, hidden]` FP32), reduced by an unweighted mean — the
//!    reference's `h.mean(dim=2)`. DFlash copies `hidden_states()`, which on
//!    the V4 path is the single-stream scratch `hc_pre` collapses into, i.e.
//!    the attention site's INPUT, not the block's output.
//! 2. **There is no rank-0 early return.** `try_dflash_capture` returns early
//!    on `comm.rank() != 0` because the DFlash drafter is replicated on rank 0.
//!    DSpark's stages are EP-SHARDED and BOTH ranks run the propose, so the
//!    worker needs the identical capture; `hc_streams` is post-all-reduce and
//!    identical on both. Skipping it on the worker is the documented
//!    `AVAROK_MTP_EP_PROPOSE` fourth-word failure — two ranks all-reducing
//!    partials of different inputs.

use anyhow::Result;

use super::TransformerModel;
use crate::layers::DsparkProposerState;
use crate::layers::deepseek_v4_dspark::{
    launch_hc_mean_capture, row_stride_elems, rows_through, slot_offset_elems,
};
use crate::speculative::ProposerState;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Tell the sequence's DSpark state how many band rows hold committed
    /// positions, right before its propose. No-op unless DSpark is active.
    ///
    /// The source is `last_mtp_hidden_idx` — the target row the scheduler
    /// pointed `save_hidden_for_mtp` at before this propose (`num_accepted`
    /// after a verify, 0 after a serial decode; the EP worker latches the same
    /// index from the `EP_CMD_MTP_PROPOSE` fourth word). Under the band law
    /// (`deepseek_v4_dspark/capture.rs`) that row is the newest committed one,
    /// so the band holds `rows_through(idx)` committed rows. `after_verify`
    /// cannot supply this: the serial-decode bootstrap never calls it.
    pub(super) fn dspark_note_committed_rows(&self, state: &mut dyn ProposerState) {
        if self.dspark_capture.is_none() {
            return;
        }
        if let Some(st) = state.as_any_mut().downcast_mut::<DsparkProposerState>() {
            let idx = self
                .last_mtp_hidden_idx
                .load(std::sync::atomic::Ordering::Relaxed);
            st.committed_rows = rows_through(idx);
        }
    }

    /// Slot of `layer_idx` in the capture band's feature axis, or `None` when
    /// this layer is not a DSpark target.
    fn dspark_slot(&self, layer_idx: usize) -> Option<usize> {
        self.dspark_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
    }

    /// Capture `k` rows of the highway starting at `src_row0` into band rows
    /// `dst_row0 .. dst_row0 + k`.
    ///
    /// Every single-sequence caller passes `src_row0 = dst_row0 = 0` and
    /// `k = rows_through(token_idx)` — the band law in
    /// `deepseek_v4_dspark/capture.rs`: band row i IS target row i, for a
    /// decode (`k = 1`) and a K-row verify (`k = K`) alike. The four
    /// coordinates exist for the cross-sequence `try_dflash_capture_all_at`
    /// form.
    pub(super) fn try_dspark_capture_rows(
        &self,
        layer_idx: usize,
        src_row0: usize,
        k: usize,
        dst_row0: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(dst) = self.dspark_capture else {
            return Ok(());
        };
        let Some(slot) = self.dspark_slot(layer_idx) else {
            return Ok(());
        };
        let n_targets = self.dspark_capture_layers.len();
        let hidden = self.config.hidden_size;
        let stride = row_stride_elems(n_targets, hidden);
        // Refuse to write past the band rather than corrupt the next
        // sequence's rows; the propose then sees fewer committed rows and
        // reports a gap, which degrades acceptance and nothing else.
        let capped = k.min(self.dspark_capture_rows.saturating_sub(dst_row0));
        if capped < k {
            tracing::debug!(
                "DSpark capture clipped: rows {dst_row0}..{} exceed capacity {}",
                dst_row0 + k,
                self.dspark_capture_rows,
            );
        }
        launch_hc_mean_capture(
            self.gpu.as_ref(),
            self.dspark_capture_k,
            self.buffers.hc_streams(),
            dst.offset(dst_row0 * stride * 2),
            capped as u32,
            hidden as u32,
            self.config.hc_mult as u32,
            src_row0 as u32,
            stride as u32,
            slot_offset_elems(slot, hidden) as u32,
            stream,
        )
    }

    /// Prefill capture: the prompt's hc-means into the sequence's own
    /// `[window, n_targets * hidden]` ring at slot `pos % window`.
    ///
    /// The ring keeps only the last `window` positions — later writes overwrite
    /// earlier slots, which IS the reference's `cutoff` split. A chunk that
    /// crosses the wrap is split into contiguous runs so the batched kernel's
    /// `dst + r * stride` addressing stays correct.
    ///
    /// 🪤 Reads `hc_streams()` at row offset 0. The V4 bring-up serves
    /// `--max-batch-size 1` with no chunk continuation, so a prefill step is
    /// never mixed with decode rows and `ctx.hc_row_offset` is 0 there; the
    /// DFlash prefill capture this branches out of makes the same assumption
    /// about `hidden_states()`.
    pub(super) fn try_dspark_prefill_capture_layer(
        &self,
        seq: &mut SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dspark_capture.is_none() {
            return Ok(());
        }
        let Some(slot) = self.dspark_slot(layer_idx) else {
            return Ok(());
        };
        let window = self.config.sliding_window as usize;
        if window == 0 {
            return Ok(());
        }
        let n_targets = self.dspark_capture_layers.len();
        let hidden = self.config.hidden_size;
        let stride = row_stride_elems(n_targets, hidden);
        let slot_off = slot_offset_elems(slot, hidden);
        let hc_mult = self.config.hc_mult as u32;
        let kernel = self.dspark_capture_k;
        let gpu = self.gpu.as_ref();
        let streams = self.buffers.hc_streams();

        let Some(state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        let Some(dstate) = state.as_any_mut().downcast_mut::<DsparkProposerState>() else {
            return Ok(());
        };

        let mut done = 0usize;
        while done < proc_count {
            let pos = chunk_start + done;
            let ring_slot = pos % window;
            let run = (proc_count - done).min(window - ring_slot);
            launch_hc_mean_capture(
                gpu,
                kernel,
                streams,
                dstate.prefill_capture_ring.offset(ring_slot * stride * 2),
                run as u32,
                hidden as u32,
                hc_mult,
                done as u32,
                stride as u32,
                slot_off as u32,
                stream,
            )?;
            done += run;
        }
        // Idempotent across chunks and across the three capture layers: both
        // are functions of the prompt length alone.
        dstate.prefill_prompt_len = chunk_start + proc_count;
        dstate.prefill_filled = dstate.prefill_prompt_len.min(window);
        Ok(())
    }
}
