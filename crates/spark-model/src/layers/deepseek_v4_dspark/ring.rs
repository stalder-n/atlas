// SPDX-License-Identifier: AGPL-3.0-only

//! Pure host index arithmetic for the DSpark KV ring.
//!
//! Nothing here touches the GPU, so every rule the propose depends on is
//! unit-testable on a CUDA-less host. The rules come from
//! `DSparkAttention.forward` in the reference (`inference/model.py`, l. 800):
//!
//! ```text
//! kv_cache[:, start_pos % window] = main_kv                 # ring write
//! topk_idxs = [arange(min(window, start_pos + 1))           # valid ring prefix
//!              ‖ window + arange(block_size)]               # this block's own KV
//! ```
//!
//! and they are IDENTICAL for every one of the γ queries — the block is
//! bidirectional over its own positions and sees the whole valid ring.
//!
//! ## Why Atlas places the block KV at `ring_len` instead of at `window`
//!
//! The reference can hand `sparse_attn` an arbitrary index list.
//! `prefill_attn_compressed_vis` takes one INCLUSIVE `[lo, hi]` interval per
//! query, so the keys it reads must be contiguous. While the ring is still
//! filling (`ring_len < window`) the reference's two ranges —
//! `[0, ring_len)` and `[window, window + γ)` — are not.
//!
//! Writing the block KV at `ring_len` makes them one range `[0, ring_len + γ)`
//! and changes nothing: softmax over keys is permutation-invariant, RoPE is
//! applied at write time (so a key carries its own position), and the slots
//! `[ring_len, ring_len + γ)` are exactly the ones the NEXT catch-up overwrites
//! (position `p` lands at `p % window`, and the next position to arrive is
//! `next_pos == ring_len` while unwrapped). Once `ring_len == window` the block
//! lands at `window`, which is the reference's own placement.

/// Ring slot of an absolute sequence position.
pub fn slot(pos: usize, window: usize) -> usize {
    debug_assert!(window > 0, "DSpark ring window must be non-zero (PCND)");
    pos % window
}

/// Valid ring entries once positions `0..next_pos` have been written.
pub fn ring_len(next_pos: usize, window: usize) -> usize {
    next_pos.min(window)
}

/// First slot of this block's own KV, in the `window + block` slot buffer.
pub fn block_base(ring_len: usize) -> usize {
    ring_len
}

/// Inclusive per-query key bounds handed to `prefill_attn_compressed_vis`.
/// Identical for all γ queries: the block is bidirectional.
pub fn key_bounds(ring_len: usize, block: usize) -> (u32, u32) {
    debug_assert!(block > 0, "DSpark block size must be non-zero (PCND)");
    (0, (ring_len + block - 1) as u32)
}

/// Slot capacity one stage's ring needs.
pub fn ring_slots(window: usize, block: usize) -> usize {
    window + block
}

/// Prompt positions whose captures seed the ring, oldest first, paired with
/// the slot each lands in.
///
/// Mirrors the reference's `cutoff` split of `Compressor`/`DSparkAttention`
/// prefill: only the last `window` prompt positions survive, and each keeps
/// its own `pos % window` slot rather than being compacted to 0.
pub fn prefill_drain(prompt_len: usize, window: usize) -> Vec<(usize, usize)> {
    let first = prompt_len.saturating_sub(window);
    (first..prompt_len).map(|p| (p, slot(p, window))).collect()
}

/// Which committed capture rows feed the ring on a warm propose.
///
/// `next_pos` is the first sequence position whose ring entry is missing;
/// `position` is the position the propose is drafting FROM (so positions
/// `next_pos..=position` are the ones still owed). `committed` is how many
/// rows the last verify left in the capture band (`num_accepted + 1`).
///
/// A `gap > 0` means the sequence advanced without a propose — serial-decode
/// bootstrap, adaptive-speculation suspend, preemption/resume, a prefix-cache
/// warm turn. The design DEGRADES there (older ring entries, lower acceptance)
/// and cannot corrupt: every draft is re-verified by the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatchUp {
    /// First absolute sequence position this catch-up writes.
    pub first_pos: usize,
    /// How many positions it writes.
    pub count: usize,
    /// First capture-band row it reads (rows are `first_row..first_row+count`).
    pub first_row: usize,
    /// Positions that are owed but have no capture row left to rebuild them.
    pub gap: usize,
}

pub fn catch_up(next_pos: usize, position: usize, committed: usize) -> CatchUp {
    let owed = (position + 1).saturating_sub(next_pos);
    let count = owed.min(committed);
    CatchUp {
        first_pos: position + 1 - count,
        count,
        first_row: committed - count,
        gap: owed - count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_is_position_modulo_window() {
        assert_eq!(slot(0, 128), 0);
        assert_eq!(slot(127, 128), 127);
        // The wrap the reference relies on: position `window` reuses slot 0.
        assert_eq!(slot(128, 128), 0);
        assert_eq!(slot(2 * 128 + 7, 128), 7);
    }

    #[test]
    fn ring_len_saturates_at_the_window() {
        assert_eq!(ring_len(0, 128), 0);
        assert_eq!(ring_len(1, 128), 1);
        assert_eq!(ring_len(127, 128), 127);
        assert_eq!(ring_len(128, 128), 128);
        assert_eq!(ring_len(100_000, 128), 128);
    }

    /// The block KV is contiguous with the valid ring prefix, and the whole
    /// key range fits the `window + block` slot buffer.
    #[test]
    fn block_kv_is_contiguous_with_the_valid_prefix() {
        let (window, block) = (128usize, 5usize);
        for next_pos in [0usize, 1, 7, 127, 128, 129, 10_000] {
            let len = ring_len(next_pos, window);
            let base = block_base(len);
            let (lo, hi) = key_bounds(len, block);
            assert_eq!(lo, 0);
            assert_eq!(hi as usize, len + block - 1);
            assert_eq!(base, len);
            assert!(
                base + block <= ring_slots(window, block),
                "block at {base}..{} overruns {} slots",
                base + block,
                ring_slots(window, block)
            );
        }
    }

    /// A prompt shorter than one window keeps every position; a longer one
    /// keeps the last `window`, each in its OWN `pos % window` slot.
    #[test]
    fn prefill_drain_keeps_the_last_window_in_place() {
        let window = 8usize;

        let short = prefill_drain(5, window);
        assert_eq!(short, vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);

        let exact = prefill_drain(window, window);
        assert_eq!(exact.len(), window);
        assert_eq!(exact.first(), Some(&(0usize, 0usize)));
        assert_eq!(exact.last(), Some(&(7usize, 7usize)));

        // P = 2*window + 7: positions 16..23 survive, wrapping to slots 0..7.
        let wrapped = prefill_drain(2 * window + 7, window);
        assert_eq!(wrapped.len(), window);
        assert_eq!(
            wrapped,
            vec![
                (15, 7),
                (16, 0),
                (17, 1),
                (18, 2),
                (19, 3),
                (20, 4),
                (21, 5),
                (22, 6)
            ]
        );
        // Every slot is written exactly once — no position silently lost.
        let mut slots: Vec<usize> = wrapped.iter().map(|&(_, s)| s).collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..window).collect::<Vec<_>>());
    }

    #[test]
    fn a_zero_length_prompt_drains_nothing() {
        assert!(prefill_drain(0, 128).is_empty());
    }

    /// K=4 verify commits `num_accepted + 1` rows; the catch-up consumes the
    /// NEWEST `owed` of them and reads them from the tail of the band.
    #[test]
    fn catch_up_consumes_the_newest_committed_rows() {
        // Steady state: one accepted token per verify, exactly one owed row.
        for accepted in 0..4usize {
            let committed = accepted + 1;
            // The propose after this verify drafts from position
            // `next_pos + accepted`.
            let next_pos = 100usize;
            let position = next_pos + accepted;
            let c = catch_up(next_pos, position, committed);
            assert_eq!(c.count, committed, "accepted={accepted}");
            assert_eq!(c.first_pos, next_pos);
            assert_eq!(c.first_row, 0);
            assert_eq!(c.gap, 0);
        }
    }

    #[test]
    fn catch_up_reports_a_gap_it_cannot_fill() {
        // Ten positions owed, only two capture rows survive: the two NEWEST
        // positions are rebuilt and eight are reported as a gap.
        let c = catch_up(90, 99, 2);
        assert_eq!(
            c,
            CatchUp {
                first_pos: 98,
                count: 2,
                first_row: 0,
                gap: 8
            }
        );
    }

    #[test]
    fn catch_up_is_empty_when_the_ring_is_already_current() {
        let c = catch_up(100, 99, 4);
        assert_eq!(c.count, 0);
        assert_eq!(c.gap, 0);
        assert_eq!(c.first_pos, 100);
        // Nothing is read, so the row cursor sits at the end of the band.
        assert_eq!(c.first_row, 4);
    }

    /// More committed rows than owed positions: only the owed tail is used.
    #[test]
    fn catch_up_never_reads_more_rows_than_it_owes() {
        let c = catch_up(100, 100, 4);
        assert_eq!(c.count, 1);
        assert_eq!(c.first_pos, 100);
        assert_eq!(c.first_row, 3);
        assert_eq!(c.gap, 0);
    }

    /// The wrap the ring must survive: the last pre-wrap position and the
    /// first post-wrap one land in slots `window-1` and `0`.
    #[test]
    fn catch_up_across_the_wrap_keeps_modular_slots() {
        let window = 128usize;
        let c = catch_up(window - 1, window, 2);
        assert_eq!(c.count, 2);
        assert_eq!(c.first_pos, window - 1);
        let slots: Vec<usize> = (c.first_pos..c.first_pos + c.count)
            .map(|p| slot(p, window))
            .collect();
        assert_eq!(slots, vec![window - 1, 0]);
        assert_eq!(ring_len(window + 1, window), window);
    }
}
