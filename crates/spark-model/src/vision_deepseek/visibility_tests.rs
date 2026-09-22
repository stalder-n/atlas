// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the image-span attention bounds.
//!
//! Expectations come from `model.py::get_image_visible` and
//! `get_window_topk_idxs_visible` at revision `86f746b3…`: for every query the generator
//! collected the reference's row of visible raw keys, asserted the set is an interval, and
//! recorded it (verbatim for short streams, as `lo`/`hi`/cardinality for long ones).

use super::*;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/visibility.json");
const REVISION: &str = "86f746b36186f0e567729a5c06a8c918caba82a9";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/visibility.json parses")
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64") as u32)
        .collect()
}

/// `(vocab_size, window, max_image_tokens)` — the parameters the goldens were generated
/// with, which every comparison below must use.
fn params(f: &Value) -> (u32, usize, usize) {
    let g = |k: &str| {
        f[k].as_u64()
            .unwrap_or_else(|| panic!("fixture is missing `{k}`"))
    };
    (
        g("vocab_size") as u32,
        g("window") as usize,
        g("max_image_tokens") as usize,
    )
}

struct Stream {
    name: String,
    tokens: Vec<u32>,
    left: Vec<u32>,
    right: Vec<u32>,
    lo: Vec<u32>,
    hi: Vec<u32>,
    counts: Vec<usize>,
    sets: Option<Vec<Vec<u32>>>,
    has_span: bool,
}

fn streams(f: &Value) -> Vec<Stream> {
    f["streams"]
        .as_array()
        .expect("streams")
        .iter()
        .map(|s| Stream {
            name: s["name"].as_str().expect("name").to_string(),
            tokens: u32s(&s["tokens"]),
            left: u32s(&s["left"]),
            right: u32s(&s["right"]),
            lo: u32s(&s["lo"]),
            hi: u32s(&s["hi"]),
            counts: s["visible_counts"]
                .as_array()
                .expect("counts")
                .iter()
                .map(|x| x.as_u64().expect("u64") as usize)
                .collect(),
            sets: s
                .get("visible_sets")
                .and_then(Value::as_array)
                .map(|rows| rows.iter().map(u32s).collect()),
            has_span: s["has_image_span"].as_bool().expect("has_image_span"),
        })
        .collect()
}

#[test]
fn golden_provenance_and_coverage() {
    let f = fixture();
    assert_eq!(f["reference"]["revision"], REVISION);
    assert_eq!(f["window"], 128);
    assert_eq!(f["max_image_tokens"], 384);
    assert_eq!(f["vocab_size"], 129_280);
    let ss = streams(&f);
    assert!(
        ss.len() >= 6,
        "the spec requires >= 6 streams, got {}",
        ss.len()
    );
    for want in [
        "text_only",
        "image_at_zero",
        "image_mid",
        "two_adjacent",
        "image_at_end",
        "prompt_2048",
        "synthetic_malformed_600",
    ] {
        assert!(
            ss.iter().any(|s| s.name == want),
            "stream `{want}` is missing"
        );
    }
    assert!(
        ss.iter().any(|s| s.tokens.len() >= 2048),
        "no stream exceeds window + max_image_tokens"
    );
    assert!(
        ss.iter().any(|s| s.sets.is_some()),
        "no stream carries the reference's verbatim visible sets"
    );
    // Every reference row is an interval — the premise that lets Atlas ship (lo, hi)
    // instead of an index matrix.
    for s in f["streams"].as_array().expect("streams") {
        assert!(
            s["visible_contiguous"].as_bool().unwrap_or(false),
            "{} is not contiguous",
            s["name"]
        );
    }
}

#[test]
fn image_visible_matches_the_reference() {
    let f = fixture();
    let (vocab, _w, m) = params(&f);
    for s in streams(&f) {
        let (left, right) = image_visible(&s.tokens, vocab, m);
        assert_eq!(left, s.left, "left for `{}`", s.name);
        assert_eq!(right, s.right, "right for `{}`", s.name);
    }
}

#[test]
fn window_bounds_match_the_reference() {
    let f = fixture();
    let (vocab, w, m) = params(&f);
    for s in streams(&f) {
        let got = window_bounds(&s.tokens, vocab, w, m);
        assert_eq!(got.lo, s.lo, "lo for `{}`", s.name);
        assert_eq!(got.hi, s.hi, "hi for `{}`", s.name);
    }
}

/// `{lo..=hi}` must be exactly the set of raw keys the reference's index matrix exposes —
/// the substitution the whole `_vis` attention kernel rests on.
#[test]
fn the_bound_interval_is_exactly_the_reference_visible_set() {
    let f = fixture();
    let (vocab, w, m) = params(&f);
    let mut verbatim = 0usize;
    for s in streams(&f) {
        let got = window_bounds(&s.tokens, vocab, w, m);
        for t in 0..s.tokens.len() {
            let (lo, hi) = (got.lo[t], got.hi[t]);
            assert!(lo <= hi, "`{}` query {t}: empty range {lo}..={hi}", s.name);
            let span = (hi - lo + 1) as usize;
            assert_eq!(span, s.counts[t], "`{}` query {t}: visible count", s.name);
            if let Some(sets) = &s.sets {
                assert_eq!(
                    (lo..=hi).collect::<Vec<u32>>(),
                    sets[t],
                    "`{}` query {t}: visible set",
                    s.name
                );
                verbatim += 1;
            }
        }
    }
    assert!(
        verbatim >= 40,
        "too few verbatim set comparisons: {verbatim}"
    );
}

/// U2.2's kernel-side assertion: `hi` is always a valid index into the chunk. This is
/// unconditional, including on the truncated/malformed stream.
#[test]
fn hi_never_leaves_the_chunk() {
    let f = fixture();
    let (vocab, w, m) = params(&f);
    for s in streams(&f) {
        let got = window_bounds(&s.tokens, vocab, w, m);
        let n = s.tokens.len() as u32;
        for t in 0..s.tokens.len() {
            assert!(
                got.hi[t] < n,
                "`{}` query {t}: hi {} >= len {n}",
                s.name,
                got.hi[t]
            );
            assert!(
                got.lo[t] <= t as u32,
                "`{}` query {t}: lo must not exceed the query",
                s.name
            );
        }
    }
}

/// Text queries keep the plain causal sliding window — a regression here silently changes
/// every text-only prompt.
#[test]
fn text_queries_keep_the_plain_causal_window() {
    let f = fixture();
    let (vocab, w, m) = params(&f);
    let s = streams(&f)
        .into_iter()
        .find(|s| s.name == "text_only")
        .expect("text_only stream");
    let got = window_bounds(&s.tokens, vocab, w, m);
    for t in 0..s.tokens.len() {
        assert_eq!(
            got.lo[t],
            (t as u32).saturating_sub(w as u32 - 1),
            "lo at {t}"
        );
        assert_eq!(
            got.hi[t], t as u32,
            "a text query must not see the future at {t}"
        );
    }
    // And the same holds for every non-sentinel query of a stream that DOES hold an image,
    // outside the span.
    let mixed = streams(&f)
        .into_iter()
        .find(|s| s.name == "image_mid")
        .expect("image_mid stream");
    let got = window_bounds(&mixed.tokens, vocab, w, m);
    for t in 0..mixed.tokens.len() {
        if mixed.left[t] == 0 && mixed.right[t] == 0 {
            assert_eq!(got.lo[t], (t as u32).saturating_sub(w as u32 - 1));
            assert_eq!(got.hi[t], t as u32, "text query {t} must stay causal");
        }
    }
}

/// The `left`/`right` clamps and the `lo + W + M - 1` cap are unreachable for any real
/// image (a block is at most 381 tokens), so the SYNTHETIC 600-token span is the only
/// thing that pins them against the reference rather than against a reading of it.
#[test]
fn the_synthetic_malformed_span_exercises_the_clamps_and_the_cap() {
    let f = fixture();
    let (vocab, w, m) = params(&f);
    let (w, m) = (w as u32, m as u32);
    let s = streams(&f)
        .into_iter()
        .find(|s| s.name == "synthetic_malformed_600")
        .expect("synthetic_malformed_600 stream");

    assert_eq!(
        *s.left.iter().max().expect("left"),
        m - 1,
        "left must clamp at M-1"
    );
    assert_eq!(
        *s.right.iter().max().expect("right"),
        m,
        "right must clamp at M"
    );
    let cap_binds = (0..s.tokens.len())
        .filter(|&t| t as u32 + s.right[t] > s.lo[t] + w + m - 1)
        .count();
    assert!(
        cap_binds > 0,
        "the golden no longer exercises the lo + W + M - 1 cap"
    );

    let (left, right) = image_visible(&s.tokens, vocab, m as usize);
    assert_eq!(left, s.left);
    assert_eq!(right, s.right);
    let got = window_bounds(&s.tokens, vocab, w as usize, m as usize);
    assert_eq!(got.lo, s.lo);
    assert_eq!(got.hi, s.hi);
}

#[test]
fn has_image_span_matches_the_stream() {
    let f = fixture();
    let (vocab, _w, _m) = params(&f);
    for s in streams(&f) {
        assert_eq!(has_image_span(&s.tokens, vocab), s.has_span, "`{}`", s.name);
    }
    assert!(!has_image_span(&[], 129_280));
    assert!(
        !has_image_span(&[129_279], 129_280),
        "the last real token is not a sentinel"
    );
    assert!(has_image_span(&[1, 2, 129_280], 129_280));
    assert!(has_image_span(&[129_284], 129_280));
}

#[test]
fn empty_and_single_token_streams_are_handled() {
    let vocab = 129_280u32;
    let (l, r) = image_visible(&[], vocab, 384);
    assert!(l.is_empty() && r.is_empty());
    let wb = window_bounds(&[], vocab, 128, 384);
    assert!(wb.lo.is_empty() && wb.hi.is_empty());
    let wb = window_bounds(&[7], vocab, 128, 384);
    assert_eq!((wb.lo, wb.hi), (vec![0], vec![0]));
}
