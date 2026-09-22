// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the ring/scheduler contract of native DSpark — split out of
//! `tests.rs` (the golden-fed chain/confidence tests) by the 500-LoC cap.

use super::{ring, rows_through};

// ── The ring/scheduler contract, as pure arithmetic ──
//
// `position` is the scheduler's `seq.seq_len`: the FIRST UNPROCESSED position
// (where `last_token` sits), one past the reference's `start_pos`. These
// tests walk the exact call sequence `propose()` makes — `prefill_drain`,
// `catch_up`, `after_propose`, `ring_len`, `block_positions`, `runs` — so a
// drift in any of them, or in the band law (`rows_through`), fails here
// rather than on the Spark as "low acceptance".

/// One propose's worth of ring bookkeeping, mirroring `refresh_rings`.
fn propose_step(
    next_pos: usize,
    position: usize,
    committed: usize,
    window: usize,
) -> (ring::CatchUp, usize, usize) {
    let plan = ring::catch_up(next_pos, position, committed);
    let next = ring::after_propose(next_pos, position);
    (plan, next, ring::ring_len(next, window))
}

/// The bring-up trace: a 10-token prompt against a 128-slot ring, γ = 5.
///
/// Propose #1 at `position = 10` (the task's cold shape, `committed = 0`):
/// the prefill left positions 0..10 in the ring, so NOTHING is owed, the
/// ring is 10 long and the block queries sit at 10..15. A K=4 verify then
/// runs rows 0..4 at positions 10..14; with one accepted draft the next
/// propose arrives at `position = 12` holding `rows_through(1) = 2`
/// committed rows, and ingests band rows 0 and 1 as positions 10 and 11.
#[test]
fn the_bring_up_trace_owes_nothing_cold_and_ingests_the_committed_rows_warm() {
    let (p, w, gamma) = (10usize, 128usize, 5usize);

    // Cold: the drain seeds positions 0..P and hands `next_pos = P` on.
    let drain = ring::prefill_drain(p, w);
    assert_eq!(drain.len(), p);
    assert_eq!(drain.last(), Some(&(9usize, 9usize)));
    let next_pos = p;

    // Propose #1 at position P, nothing committed since the prefill.
    let (plan, next_pos, ring_len) = propose_step(next_pos, p, 0, w);
    assert_eq!((plan.count, plan.gap), (0, 0), "cold propose owes nothing");
    assert_eq!(next_pos, p);
    assert_eq!(ring_len, p, "slot P is NOT counted: nothing wrote it");
    assert_eq!(ring::block_positions(p, gamma), 10..15);
    assert_eq!(ring::block_base(ring_len), p);
    assert_eq!(ring::key_bounds(ring_len, gamma), (0, 14));

    // Verify K=4: rows 0..4 at positions 10..14; band row i = target row i.
    let k = 4usize;
    let written = 0..rows_through(k - 1);
    assert_eq!(written, 0..4);
    // Accept a = 1: positions 10 and 11 are committed; the bonus generator
    // is row 1, which is where `save_hidden_for_mtp` points.
    let num_accepted = 1usize;
    let committed = rows_through(num_accepted);
    assert_eq!(committed, 2);

    // Propose #2 at position 12.
    let (plan, next_pos, ring_len) = propose_step(next_pos, p + committed, committed, w);
    assert_eq!(
        plan,
        ring::CatchUp {
            first_pos: 10,
            count: 2,
            first_row: 0,
            gap: 0
        }
    );
    assert!(
        written.contains(&plan.first_row) && written.contains(&(plan.first_row + plan.count - 1))
    );
    assert_eq!(ring::runs(plan.first_pos, plan.count, w), vec![(10, 10, 2)]);
    assert_eq!(next_pos, 12);
    assert_eq!(ring_len, 12);
    assert_eq!(ring::block_positions(12, gamma), 12..17);
}

/// The path the scheduler actually serves (`dflash_verify_raw_argmax` is
/// false on a `--speculative` serve): the first propose follows ONE serial
/// decode. That decode captured position P into band row 0 and pointed
/// `save_hidden_for_mtp` at row 0, so the propose at P+1 ingests it and owes
/// nothing — no cold-turn gap.
#[test]
fn the_served_bootstrap_ingests_the_serial_decode_and_leaves_no_gap() {
    let (p, w) = (10usize, 128usize);
    let next_pos = p; // after the drain
    let committed = rows_through(0);
    let (plan, next_pos, ring_len) = propose_step(next_pos, p + 1, committed, w);
    assert_eq!(
        plan,
        ring::CatchUp {
            first_pos: p,
            count: 1,
            first_row: 0,
            gap: 0
        }
    );
    assert_eq!((next_pos, ring_len), (p + 1, p + 1));
}

/// The band law against the catch-up: for every reachable verify width and
/// every accept count, the rows the catch-up reads were written by that
/// verify, they are the OLDEST rows (the committed prefix), and they land at
/// the positions the verify ran them at.
#[test]
fn a_k_row_verify_writes_every_row_the_catch_up_reads() {
    let (p, w) = (1000usize, 128usize);
    for k in 2..=4usize {
        let written = 0..rows_through(k - 1);
        for num_accepted in 0..k {
            let committed = rows_through(num_accepted);
            let (plan, next_pos, _) = propose_step(p, p + committed, committed, w);
            assert_eq!(plan.gap, 0, "K={k} a={num_accepted}");
            assert_eq!(plan.count, committed);
            assert_eq!(plan.first_row, 0, "the committed prefix is the OLDEST rows");
            assert!(
                written.contains(&(plan.first_row + plan.count - 1)),
                "K={k} a={num_accepted}: row {} was never written",
                plan.first_row + plan.count - 1
            );
            // Band row i was target row i, run at position p + i.
            assert_eq!(plan.first_pos, p);
            assert_eq!(next_pos, p + committed);
        }
    }

    // Several serial decodes in a row (adaptive-speculation suspend): each
    // rewrote row 0, so only the NEWEST position is rebuilt and the rest is a
    // counted gap — never a read of a row nothing wrote.
    let (plan, _, _) = propose_step(p, p + 3, rows_through(0), w);
    assert_eq!(
        plan,
        ring::CatchUp {
            first_pos: p + 2,
            count: 1,
            first_row: 0,
            gap: 2
        }
    );
}

/// A prompt longer than the window, and a warm catch-up whose committed
/// positions straddle the wrap: the drain keeps the last `window` positions
/// in their own slots, the catch-up is split into two contiguous runs, and
/// `ring_len` saturates so the block lands at `window`.
#[test]
fn the_ring_survives_the_wrap_on_both_the_drain_and_a_warm_catch_up() {
    let (w, gamma) = (128usize, 5usize);

    // Drain: P = 2W + 7 keeps positions 135..263 at slots 7..128, 0..7.
    let p = 2 * w + 7;
    let drain = ring::prefill_drain(p, w);
    assert_eq!(drain.len(), w);
    assert_eq!(drain.first(), Some(&(p - w, 7usize)));
    assert_eq!(drain.last(), Some(&(p - 1, 6usize)));
    let (plan, next_pos, ring_len) = propose_step(p, p, 0, w);
    assert_eq!((plan.count, plan.gap), (0, 0));
    assert_eq!(ring_len, w);
    assert_eq!(ring::block_base(ring_len), w);
    assert_eq!(
        ring::key_bounds(ring_len, gamma),
        (0, (w + gamma - 1) as u32)
    );
    assert_eq!(next_pos, p);

    // Warm: a K=4 full accept committing positions 126..130 across the wrap.
    let p = w - 2;
    let committed = rows_through(3);
    let (plan, next_pos, ring_len) = propose_step(p, p + committed, committed, w);
    assert_eq!(plan.first_pos, p);
    assert_eq!(plan.count, 4);
    assert_eq!(
        ring::runs(plan.first_pos, plan.count, w),
        vec![(126, 126, 2), (128, 0, 2)],
        "the run is cut at the wrap so band rows 2,3 land in slots 0,1"
    );
    assert_eq!(next_pos, w + 2);
    assert_eq!(ring_len, w);
    assert_eq!(ring::block_positions(next_pos, gamma), 130..135);
}
