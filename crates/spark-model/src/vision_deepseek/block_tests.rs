// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the image-BLOCK port — the SSOT of the DeepSeek image-token layout.
//!
//! Expectations come from `inference/image_processor.py::build_image_block` and the
//! expansion loop of `prepare_vl_inputs`, run at revision `86f746b3…` by
//! `gen_layout_goldens.py` into `fixtures/block.json`.

use super::*;
use crate::vision_deepseek::geometry::COMPRESS_PAD_TO;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/block.json");
const REVISION: &str = "86f746b36186f0e567729a5c06a8c918caba82a9";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/block.json parses")
}

fn u(v: &Value, k: &str) -> usize {
    v.get(k)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("golden row is missing `{k}`: {v}")) as usize
}

fn rows<'a>(f: &'a Value, k: &str) -> &'a Vec<Value> {
    f[k].as_array().unwrap_or_else(|| panic!("no `{k}` array"))
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64") as u32)
        .collect()
}

/// The golden encodes each block's type sequence as one decimal digit per token.
fn digits(v: &Value) -> Vec<u8> {
    v.as_str()
        .expect("types is a digit string")
        .bytes()
        .map(|c| {
            assert!((b'0'..=b'4').contains(&c), "type digit out of range: {c}");
            c - b'0'
        })
        .collect()
}

#[test]
fn golden_provenance_and_coverage() {
    let f = fixture();
    assert_eq!(f["reference"]["revision"], REVISION);
    assert_eq!(u(&f, "vocab_size"), 129_280);
    assert_eq!(u(&f, "compress_pad_to"), COMPRESS_PAD_TO);
    let grids = rows(&f, "grids");
    assert_eq!(
        grids.len(),
        144,
        "the spec requires the full 12x12 grid sweep"
    );
    for g in grids {
        assert_eq!(
            g["types"].as_object().expect("types map").len(),
            4,
            "each grid needs all four start_pos residues"
        );
    }
    assert!(rows(&f, "large_start_pos").len() >= 12);
    assert!(rows(&f, "expansions").len() >= 5);
}

/// All 576 `(n_llm_h ∈ 1..=12, n_llm_w ∈ 1..=12, start_pos mod 4)` cases, types and perm.
#[test]
fn build_image_block_matches_the_reference_for_every_grid() {
    let f = fixture();
    let mut cases = 0usize;
    for g in rows(&f, "grids") {
        let (nh, nw) = (u(g, "h"), u(g, "w"));
        let want_perm = u32s(&g["perm"]);
        for start_pos in 0..4usize {
            let want_types = digits(&g["types"][start_pos.to_string()]);
            let got = build_image_block(nh, nw, start_pos);
            assert_eq!(got.types, want_types, "types for ({nh},{nw}) @ {start_pos}");
            assert_eq!(got.perm, want_perm, "perm for ({nh},{nw}) @ {start_pos}");
            cases += 1;
        }
    }
    assert_eq!(cases, 576);
}

/// Only `start_pos mod 4` matters — the reference's `compress_pad` is the only place
/// `start_pos` appears.
#[test]
fn only_the_start_pos_residue_matters() {
    let f = fixture();
    let mut cases = 0usize;
    for row in rows(&f, "large_start_pos") {
        let (nh, nw, sp) = (u(row, "h"), u(row, "w"), u(row, "start_pos"));
        let got = build_image_block(nh, nw, sp);
        assert_eq!(got.types, digits(&row["types"]), "types ({nh},{nw}) @ {sp}");
        assert_eq!(got.perm, u32s(&row["perm"]), "perm ({nh},{nw}) @ {sp}");
        assert_eq!(
            got.types,
            build_image_block(nh, nw, sp % COMPRESS_PAD_TO).types,
            "({nh},{nw}): start_pos {sp} must behave like {}",
            sp % COMPRESS_PAD_TO
        );
        cases += 1;
    }
    assert!(cases >= 12);
}

/// Structural invariants the splice and the attention bounds both rely on.
#[test]
fn block_structure_invariants_hold_for_every_grid() {
    let f = fixture();
    for g in rows(&f, "grids") {
        let (nh, nw) = (u(g, "h"), u(g, "w"));
        for start_pos in 0..4usize {
            let blk = build_image_block(nh, nw, start_pos);
            let compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO;

            // Leading compress pad, then exactly one START; exactly one END, and it is last.
            assert!(
                blk.types[..compress_pad].iter().all(|&t| t == IMAGE_PAD),
                "({nh},{nw})@{start_pos}: the leading {compress_pad} tokens must be IMAGE_PAD"
            );
            assert_eq!(blk.types[compress_pad], IMAGE_START);
            assert_eq!(*blk.types.last().expect("non-empty"), IMAGE_END);
            assert_eq!(blk.types.iter().filter(|&&t| t == IMAGE_START).count(), 1);
            assert_eq!(blk.types.iter().filter(|&&t| t == IMAGE_END).count(), 1);

            // One IMAGE slot per aligner row, one NEWLINE per grid row.
            assert_eq!(
                blk.types.iter().filter(|&&t| t == IMAGE).count(),
                nh * nw,
                "({nh},{nw})@{start_pos}: IMAGE slot count"
            );
            assert_eq!(
                blk.types.iter().filter(|&&t| t == IMAGE_NEWLINE).count(),
                nh,
                "({nh},{nw})@{start_pos}: one NEWLINE per grid row"
            );

            // `perm` is a permutation of 0..nh*nw, so every aligner row is placed once.
            assert_eq!(blk.perm.len(), nh * nw);
            let mut sorted = blk.perm.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..(nh * nw) as u32).collect::<Vec<_>>(),
                "({nh},{nw}): perm must be a permutation of the aligner rows"
            );

            // START(+1) .. END is one contiguous span; nothing after END.
            assert_eq!(blk.types.len(), block_len(nh, nw, start_pos));
            assert_eq!(
                block_len(nh, nw, start_pos),
                u(g, "num_tokens") + compress_pad,
                "({nh},{nw})@{start_pos}: block_len vs grid_tokens + compress_pad"
            );
        }
        assert_eq!(perm_only(nh, nw), build_image_block(nh, nw, 0).perm);
        assert_eq!(perm_only(nh, nw), build_image_block(nh, nw, 3).perm);
    }
}

/// The row-pair interleave is what makes ratio-4 KV compression group 2x2 spatial
/// neighbours: consecutive `perm` pairs must be vertically adjacent grid cells.
#[test]
fn perm_interleaves_row_pairs() {
    let f = fixture();
    for g in rows(&f, "grids") {
        let (nh, nw) = (u(g, "h"), u(g, "w"));
        if nh < 2 {
            continue;
        }
        let perm = perm_only(nh, nw);
        // The first two entries are (row 0, col 0) and (row 1, col 0).
        assert_eq!(perm[0], 0, "({nh},{nw}): first IMAGE slot is grid cell 0");
        assert_eq!(
            perm[1], nw as u32,
            "({nh},{nw}): the second IMAGE slot must be the cell directly BELOW the first"
        );
    }
}

#[test]
fn sentinel_ids_round_trip_and_respect_the_vocab_boundary() {
    let vocab = 129_280u32;
    for ty in [IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEWLINE, IMAGE_END] {
        let id = sentinel_id(vocab, ty);
        assert_eq!(id, vocab + ty as u32);
        assert_eq!(sentinel_type(vocab, id), Some(ty));
        assert!(is_sentinel(vocab, id));
    }
    assert_eq!(sentinel_type(vocab, vocab - 1), None, "last real token");
    assert!(!is_sentinel(vocab, vocab - 1));
    assert_eq!(sentinel_type(vocab, 0), None);
    assert_eq!(sentinel_type(vocab, vocab), Some(IMAGE_START));
    assert_eq!(sentinel_type(vocab, vocab + 4), Some(IMAGE_END));
    assert_eq!(
        sentinel_type(vocab, vocab + 5),
        None,
        "one past the last sentinel"
    );
    assert!(!is_sentinel(vocab, vocab + 5));
    assert_eq!(NUM_SENTINEL_TYPES, 5);
}

/// The expansion loop: `start_pos` is the running output length, so each block's compress
/// pad depends on everything emitted before it.
#[test]
fn expand_placeholders_matches_the_reference() {
    let f = fixture();
    let vocab = u(&f, "vocab_size") as u32;
    for case in rows(&f, "expansions") {
        let note = case["note"].as_str().unwrap_or("");
        let tokens = u32s(&case["tokens"]);
        let placeholder = u(case, "placeholder") as u32;
        let grids: Vec<(usize, usize)> = case["grids"]
            .as_array()
            .expect("grids")
            .iter()
            .map(|g| {
                let g = g.as_array().expect("pair");
                (
                    g[0].as_u64().expect("h") as usize,
                    g[1].as_u64().expect("w") as usize,
                )
            })
            .collect();

        let (got_tokens, got_blocks) = expand_placeholders(&tokens, placeholder, vocab, &grids)
            .unwrap_or_else(|e| panic!("expand_placeholders [{note}]: {e:#}"));
        assert_eq!(
            got_tokens,
            u32s(&case["expanded"]),
            "expanded stream [{note}]"
        );

        let want_blocks = rows(case, "blocks");
        assert_eq!(got_blocks.len(), want_blocks.len(), "block count [{note}]");
        for (got, want) in got_blocks.iter().zip(want_blocks) {
            assert_eq!(got.types, digits(&want["types"]), "block types [{note}]");
            assert_eq!(got.perm, u32s(&want["perm"]), "block perm [{note}]");
            assert_eq!(
                got.types.len(),
                block_len(u(want, "h"), u(want, "w"), u(want, "start")),
                "block_len agrees with the emitted block [{note}]"
            );
        }

        // Every emitted sentinel is exactly `vocab + type`, and no real token was touched.
        let text_len: usize = tokens.iter().filter(|&&t| t != placeholder).count();
        let emitted: usize = got_blocks.iter().map(|b| b.types.len()).sum();
        assert_eq!(
            got_tokens.len(),
            text_len + emitted,
            "total length [{note}]"
        );
        assert_eq!(
            got_tokens.iter().filter(|&&t| t < vocab).count(),
            text_len,
            "text tokens must survive verbatim [{note}]"
        );
    }
}

/// The count mismatch is what U2.4 turns into a 400, so it must be an `Err` both ways and
/// must name both counts.
#[test]
fn expand_placeholders_rejects_a_count_mismatch() {
    let vocab = 129_280u32;
    let tokens = [10u32, 999, 11, 999, 12];
    let too_few = expand_placeholders(&tokens, 999, vocab, &[(2, 3)]);
    let too_many = expand_placeholders(&tokens, 999, vocab, &[(2, 3), (4, 4), (1, 1)]);
    for (r, what) in [
        (too_few, "1 image for 2 placeholders"),
        (too_many, "3 for 2"),
    ] {
        let e = r.expect_err(what).to_string();
        assert!(
            e.contains('2'),
            "the error must name the placeholder count: {e}"
        );
    }
    // Zero placeholders and zero grids is not an error.
    assert!(expand_placeholders(&[1, 2, 3], 999, vocab, &[]).is_ok());
}

/// The expansion never emits an id that could be sampled, and never overflows.
#[test]
fn expand_placeholders_guards_the_vocab_ceiling() {
    let r = expand_placeholders(&[999], 999, u32::MAX - 2, &[(1, 1)]);
    assert!(r.is_err(), "vocab_size + 4 must not wrap u32");
}
