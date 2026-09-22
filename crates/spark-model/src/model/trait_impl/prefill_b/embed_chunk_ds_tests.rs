// SPDX-License-Identifier: AGPL-3.0-only

//! Overlay-plan tests.
//!
//! The expectation is re-derived from `block::expand_placeholders`' own
//! output — the blocks the SERVER built to expand the prompt — rather than
//! from a hand-written token list, so a change to the layout moves both sides
//! together or fails loudly.

use super::{DsOverlaySrc, plan_overlay};
use crate::vision_deepseek::block;

const VOCAB: u32 = 100;
const PH: u32 = 7;

/// The plan an independent walk of the expansion's own blocks implies.
fn expected(
    tokens: &[u32],
    blocks: &[block::DsImageBlock],
    grids: &[(usize, usize)],
) -> Vec<(usize, DsOverlaySrc)> {
    let mut out = Vec::new();
    let (mut pos, mut row_base, mut bi) = (0usize, 0usize, 0usize);
    for &t in tokens {
        if t != PH {
            pos += 1;
            continue;
        }
        let (h, w) = grids[bi];
        let perm = block::perm_only(h, w);
        let mut j = 0usize;
        for &ty in &blocks[bi].types {
            if ty == block::IMAGE {
                out.push((pos, DsOverlaySrc::Row(row_base + perm[j] as usize)));
                j += 1;
            } else {
                out.push((pos, DsOverlaySrc::Param(ty)));
            }
            pos += 1;
        }
        assert_eq!(j, perm.len(), "block {bi} IMAGE slots vs permutation");
        row_base += h * w;
        bi += 1;
    }
    out
}

fn two_image_prompt() -> (
    Vec<u32>,
    Vec<u32>,
    Vec<block::DsImageBlock>,
    Vec<(usize, usize)>,
) {
    // Text before, between and after, so the compress pads differ per image.
    let raw = vec![11, 12, PH, 13, PH, 14, 15];
    let grids = vec![(2usize, 3usize), (4usize, 4usize)];
    let (expanded, blocks) =
        block::expand_placeholders(&raw, PH, VOCAB, &grids).expect("expansion");
    (raw, expanded, blocks, grids)
}

#[test]
fn every_sentinel_position_maps_to_the_source_the_reference_layout_implies() {
    let (raw, expanded, blocks, grids) = two_image_prompt();
    let plan = plan_overlay(&expanded, 0, expanded.len(), VOCAB, &grids).expect("plan");
    assert_eq!(plan, expected(&raw, &blocks, &grids));

    // Non-vacuous: the plan is neither empty nor the whole prompt.
    assert!(!plan.is_empty());
    assert!(
        plan.len() < expanded.len(),
        "text positions must be untouched"
    );

    // The IMAGE slots of image 0 draw from rows [0, 6) and of image 1 from
    // [6, 22) — the stage packing, in image order.
    let rows: Vec<usize> = plan
        .iter()
        .filter_map(|(_, s)| match s {
            DsOverlaySrc::Row(r) => Some(*r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 2 * 3 + 4 * 4);
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..22).collect::<Vec<_>>(),
        "every row used exactly once"
    );
    assert!(rows[..6].iter().all(|&r| r < 6), "image 0 rows come first");
}

#[test]
fn the_leading_compress_pad_takes_the_image_pad_vector() {
    let (_, expanded, _, grids) = two_image_prompt();
    let plan = plan_overlay(&expanded, 0, expanded.len(), VOCAB, &grids).expect("plan");
    // `expand_placeholders` puts image 0 at output position 2, and
    // COMPRESS_PAD_TO - 1 - 2 % 4 = 1 pad token precedes its START.
    assert_eq!(plan[0], (2, DsOverlaySrc::Param(block::IMAGE_PAD)));
    assert_eq!(plan[1], (3, DsOverlaySrc::Param(block::IMAGE_START)));
    // …and the block closes on an END.
    let last_image_0 = plan
        .iter()
        .position(|(_, s)| *s == DsOverlaySrc::Param(block::IMAGE_END))
        .expect("an END");
    assert!(last_image_0 > 1);
}

#[test]
fn a_later_chunk_window_is_the_tail_of_the_full_plan() {
    let (_, expanded, _, grids) = two_image_prompt();
    let full = plan_overlay(&expanded, 0, expanded.len(), VOCAB, &grids).expect("full");
    for chunk_start in [1usize, 5, 12, 30] {
        let chunk_len = expanded.len() - chunk_start;
        let tail = plan_overlay(&expanded, chunk_start, chunk_len, VOCAB, &grids);
        // A window that cuts an image span open is refused, not silently
        // mis-planned; otherwise it must equal the shifted tail.
        let Ok(tail) = tail else { continue };
        let want: Vec<(usize, DsOverlaySrc)> = full
            .iter()
            .filter(|(p, _)| *p >= chunk_start)
            .map(|(p, s)| (p - chunk_start, *s))
            .collect();
        assert_eq!(tail, want, "chunk_start {chunk_start}");
    }
}

#[test]
fn a_prompt_with_more_spans_than_staged_images_is_refused() {
    let (_, expanded, _, _) = two_image_prompt();
    let err = plan_overlay(&expanded, 0, expanded.len(), VOCAB, &[(2, 3)])
        .expect_err("two spans, one staged grid");
    assert!(format!("{err:#}").contains("more image spans"), "{err:#}");
}

#[test]
fn a_prompt_with_fewer_spans_than_staged_images_is_refused() {
    let grids = vec![(2usize, 3usize)];
    let (expanded, _) = block::expand_placeholders(&[11, PH, 12], PH, VOCAB, &grids).unwrap();
    let err = plan_overlay(&expanded, 0, expanded.len(), VOCAB, &[(2, 3), (4, 4)])
        .expect_err("one span, two staged grids");
    assert!(format!("{err:#}").contains("but 2 staged"), "{err:#}");
}

#[test]
fn a_grid_that_disagrees_with_the_expanded_block_is_refused() {
    let grids = vec![(2usize, 3usize)];
    let (expanded, _) = block::expand_placeholders(&[11, PH], PH, VOCAB, &grids).unwrap();
    // A wider grid supplies more permutation rows than the block has IMAGE
    // slots, so the span closes early.
    let err =
        plan_overlay(&expanded, 0, expanded.len(), VOCAB, &[(4, 4)]).expect_err("grid mismatch");
    assert!(format!("{err:#}").contains("IMAGE slots"), "{err:#}");
}

#[test]
fn a_span_cut_by_the_chunk_boundary_is_refused_rather_than_truncated() {
    let grids = vec![(2usize, 3usize)];
    let (expanded, _) = block::expand_placeholders(&[11, PH, 12], PH, VOCAB, &grids).unwrap();
    // Stop one token before the END.
    let cut = expanded.len() - 2;
    let err = plan_overlay(&expanded, 0, cut, VOCAB, &grids).expect_err("a half-prefilled span");
    assert!(format!("{err:#}").contains("single chunk"), "{err:#}");
}

#[test]
fn a_text_only_prompt_plans_nothing() {
    let plan = plan_overlay(&[1, 2, 3, 4], 0, 4, VOCAB, &[]).expect("plan");
    assert!(plan.is_empty());
}
