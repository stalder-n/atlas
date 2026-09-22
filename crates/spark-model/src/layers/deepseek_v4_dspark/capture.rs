// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark target capture: the hc-stream MEAN of the post-layer residual at
//! each `dspark_target_layer_ids` entry, scattered into a
//! `[rows, n_targets * hidden]` BF16 band.
//!
//! Reference (`inference/model.py`, `Transformer.forward`, l. 986):
//!
//! ```text
//! for i, layer in enumerate(self.layers):
//!     h = layer(...)                              # [T, hc_mult, dim] FP32 highway
//!     if i in self.dspark_target_layer_ids:
//!         main_hiddens.append(h.mean(dim=2))      # UNWEIGHTED mean over streams
//! main_hidden = torch.cat(main_hiddens, dim=-1)   # [T, n_targets * dim]
//! ```
//!
//! 🔴 The source is `ctx.buffers.hc_streams()`, NOT `hidden_states()`. On the
//! V4 path `hidden_states` is the single-stream scratch that `hc_pre` collapses
//! into — reading it would capture the ATTENTION site's input, not the block's
//! output. DFlash captures `hidden_states` because its target has no highway.
//!
//! The kernel is `dspark_capture::dspark_hc_mean_capture` (V4 target only, no
//! fallback). The learned `hyper_connection::hc_head` in the same target is the
//! WEIGHTED Sinkhorn collapse and is a different function; GLM's unweighted
//! `glm5next_hc_head` lives in another model target's module.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Threads per block in `dspark_hc_mean_capture` (mirrors the `.cu`).
const CAPTURE_BLOCK: u32 = 256;

/// Byte stride of one capture row: the `n_targets` hc-means concatenated on
/// the feature axis, matching the reference's `torch.cat(.., dim=-1)` and
/// `main_proj`'s `[hidden, hidden * n_targets]` shape.
pub fn row_stride_elems(n_targets: usize, hidden: usize) -> usize {
    n_targets * hidden
}

/// Element offset of one capture layer's slice inside a row.
pub fn slot_offset_elems(slot: usize, hidden: usize) -> usize {
    slot * hidden
}

/// Row index of the `k`-th verify row of sequence `seq_idx` in a band-major
/// capture buffer, or `None` when it would run past `capacity`.
///
/// Same band law as `dflash_hidden_save`: sequence `i` owns rows
/// `[i * kgamma, i * kgamma + kgamma)`, and single-sequence paths use band 0.
pub fn band_row(seq_idx: usize, k: usize, kgamma: usize, capacity: usize) -> Option<usize> {
    if k >= kgamma {
        return None;
    }
    let row = seq_idx.checked_mul(kgamma)?.checked_add(k)?;
    (row < capacity).then_some(row)
}

/// Launch the batched hc-stream mean into `dst`.
///
/// `rows` source rows starting at `src_row0` of the `[*, hc_mult, hidden]`
/// FP32 highway are reduced and written at
/// `dst[r * row_stride + slot_off + d]`, in BF16. ONE launch per (capture
/// layer, chunk) — the DFlash capture it mirrors issues `rows` strided D2D
/// copies, which at a 2048-token prefill × 3 layers is ~6k launches.
#[allow(clippy::too_many_arguments)]
pub fn launch_hc_mean_capture(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    dst: DevicePtr,
    rows: u32,
    hidden: u32,
    hc_mult: u32,
    src_row0: u32,
    dst_row_stride_elems: u32,
    dst_slot_off_elems: u32,
    stream: u64,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    anyhow::ensure!(
        kernel.0 != 0,
        "DSpark capture needs `dspark_capture::dspark_hc_mean_capture`, which did not resolve; \
         there is no fallback — the drafter would read an uninitialised band"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([CAPTURE_BLOCK, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(dst)
        .arg_u32(rows)
        .arg_u32(hidden)
        .arg_u32(hc_mult)
        .arg_u32(src_row0)
        .arg_u32(dst_row_stride_elems)
        .arg_u32(dst_slot_off_elems)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_is_the_targets_concatenated_on_the_feature_axis() {
        let (hidden, n_targets) = (4096usize, 3usize);
        assert_eq!(row_stride_elems(n_targets, hidden), 12288);
        // `main_proj` is [hidden, hidden * n_targets] — the row stride IS its
        // input width, which is what makes the capture band a valid GEMM input.
        for slot in 0..n_targets {
            assert_eq!(slot_offset_elems(slot, hidden), slot * hidden);
        }
        assert_eq!(
            slot_offset_elems(n_targets - 1, hidden) + hidden,
            row_stride_elems(n_targets, hidden)
        );
    }

    #[test]
    fn band_rows_are_kgamma_wide_per_sequence() {
        let (kgamma, batch) = (4usize, 8usize);
        let capacity = kgamma * batch;
        for seq in 0..batch {
            for k in 0..kgamma {
                assert_eq!(band_row(seq, k, kgamma, capacity), Some(seq * kgamma + k));
            }
            // One past the band belongs to the NEXT sequence, never to this one.
            assert_eq!(band_row(seq, kgamma, kgamma, capacity), None);
        }
    }

    #[test]
    fn band_rows_refuse_to_write_past_capacity() {
        // A batch-8 band law against a band-2 buffer: sequence 2 onwards is out.
        let (kgamma, capacity) = (4usize, 8usize);
        assert_eq!(band_row(1, 3, kgamma, capacity), Some(7));
        assert_eq!(band_row(2, 0, kgamma, capacity), None);
        assert_eq!(band_row(7, 0, kgamma, capacity), None);
    }

    #[test]
    fn k_of_one_is_the_single_sequence_decode_band() {
        for kgamma in 1..=4usize {
            assert_eq!(band_row(0, 0, kgamma, kgamma), Some(0));
        }
    }
}
