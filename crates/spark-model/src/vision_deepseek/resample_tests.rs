// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the Pillow-parity resampler.
//!
//! Expectations are the pixels Pillow 12.3 itself produced for
//! `Image.resize((dw, dh), BICUBIC)` and `ImageOps.pad(..., color=(127,127,127))` on the
//! deterministic gradient image, recorded by `gen_layout_goldens.py` into
//! `fixtures/resample.json`. Tolerance is max abs error <= 1 (u8) per the spec, and every
//! test reports the OBSERVED max error so a systematic 1-LSB bias is visible rather than
//! silently accepted.

use super::*;
use crate::vision_deepseek::testutil::gradient_rgb;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/resample.json");
const TOLERANCE: i32 = 1;

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/resample.json parses")
}

fn u(v: &Value, k: &str) -> usize {
    v.get(k)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("golden row is missing `{k}`: {v}")) as usize
}

fn bytes(v: &Value) -> Vec<u8> {
    v.as_array()
        .expect("rgb array")
        .iter()
        .map(|x| {
            let n = x.as_u64().expect("u8");
            assert!(n <= 255, "pixel out of range: {n}");
            n as u8
        })
        .collect()
}

fn rows<'a>(f: &'a Value, k: &str) -> &'a Vec<Value> {
    f[k].as_array().unwrap_or_else(|| panic!("no `{k}` array"))
}

/// Compares against the golden and returns the observed max abs error.
fn compare(got: &[u8], want: &[u8], what: &str) -> i32 {
    assert_eq!(got.len(), want.len(), "{what}: byte count");
    let mut worst = 0i32;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g as i32 - w as i32).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= TOLERANCE,
        "{what}: max abs error {worst} > {TOLERANCE} at byte {worst_at} (got {}, want {})",
        got[worst_at],
        want[worst_at]
    );
    worst
}

#[test]
fn golden_provenance_and_coverage() {
    let f = fixture();
    assert_eq!(
        f["reference"]["revision"],
        "86f746b36186f0e567729a5c06a8c918caba82a9"
    );
    assert_eq!(
        f["source"], "gradient_rgb(x,y) = ((7x+13y)%256, (x*x+3y)%256, (x^y)%256)",
        "the Rust `testutil::gradient_rgb` must match the generator's formula"
    );
    assert_eq!(bytes(&f["fill"]), vec![127u8, 127, 127]);
    assert!(
        rows(&f, "resize").len() >= 6,
        "spec requires >= 6 resize pairs"
    );
    assert!(rows(&f, "pad").len() >= 6, "spec requires >= 6 pad pairs");
    // The coverage that makes the tolerance meaningful, per the approach's risk note.
    let rz = rows(&f, "resize");
    assert!(
        rz.iter()
            .any(|r| u(r, "sw") == u(r, "dw") && u(r, "sh") != u(r, "dh")),
        "no width-equal case, so a wrongly-run horizontal pass would go unseen"
    );
    assert!(
        rz.iter()
            .any(|r| u(r, "sh") == u(r, "dh") && u(r, "sw") != u(r, "dw")),
        "no height-equal case"
    );
    assert!(
        rz.iter()
            .any(|r| u(r, "sw") == u(r, "dw") && u(r, "sh") == u(r, "dh")),
        "no identity case, so PIL's copy shortcut is untested"
    );
    assert!(
        rz.iter()
            .any(|r| u(r, "sw") >= 10 * u(r, "dw") && u(r, "sh") >= 10 * u(r, "dh")),
        "no extreme downscale, so the support-widening term is untested"
    );
    assert!(
        rz.iter()
            .any(|r| u(r, "dw") > u(r, "sw") && u(r, "dh") > u(r, "sh")),
        "no upscale case"
    );
}

/// The gradient source must be bit-identical to the generator's, or every comparison below
/// is against the wrong image.
#[test]
fn gradient_source_matches_the_generator() {
    let img = gradient_rgb(4, 3);
    assert_eq!(img.len(), 4 * 3 * 3);
    // (x=0,y=0) -> (0, 0, 0); (x=3,y=2) -> (7*3+13*2=47, 9+6=15, 3^2=1)
    assert_eq!(&img[0..3], &[0, 0, 0]);
    let i = (2 * 4 + 3) * 3;
    assert_eq!(&img[i..i + 3], &[47, 15, 1]);
    // (x=1,y=0) -> (7, 1, 1)
    assert_eq!(&img[3..6], &[7, 1, 1]);
}

#[test]
fn resize_matches_pillow_bicubic() {
    let f = fixture();
    let mut worst_overall = 0i32;
    for row in rows(&f, "resize") {
        let (sw, sh, dw, dh) = (u(row, "sw"), u(row, "sh"), u(row, "dw"), u(row, "dh"));
        let note = row["note"].as_str().unwrap_or("");
        let src = gradient_rgb(sw, sh);
        let got = resize_pil_bicubic_rgb(&src, sw, sh, dw, dh);
        assert_eq!(
            got.len(),
            dw * dh * 3,
            "resize {sw}x{sh}->{dw}x{dh} [{note}] size"
        );
        let worst = compare(
            &got,
            &bytes(&row["rgb"]),
            &format!("resize {sw}x{sh}->{dw}x{dh} [{note}]"),
        );
        worst_overall = worst_overall.max(worst);
    }
    println!("resize: worst observed abs error = {worst_overall}");
}

/// PIL's `Image.resize` returns a copy when the size is unchanged — byte-exact, no filter.
#[test]
fn identity_resize_is_byte_exact() {
    let src = gradient_rgb(28, 28);
    assert_eq!(resize_pil_bicubic_rgb(&src, 28, 28, 28, 28), src);
}

#[test]
fn pad_matches_pillow_imageops_pad() {
    let f = fixture();
    let mut worst_overall = 0i32;
    for row in rows(&f, "pad") {
        let (sw, sh, dw, dh) = (u(row, "sw"), u(row, "sh"), u(row, "dw"), u(row, "dh"));
        let note = row["note"].as_str().unwrap_or("");
        let src = gradient_rgb(sw, sh);
        let got = pad_contain_gray(&src, sw, sh, dw, dh);
        assert_eq!(
            got.len(),
            dw * dh * 3,
            "pad {sw}x{sh}->{dw}x{dh} [{note}] size"
        );
        let worst = compare(
            &got,
            &bytes(&row["rgb"]),
            &format!("pad {sw}x{sh}->{dw}x{dh} [{note}]"),
        );
        worst_overall = worst_overall.max(worst);
    }
    println!("pad: worst observed abs error = {worst_overall}");
}

/// The fill is exactly (127, 127, 127) — not 128, and not interpolated at the seam.
#[test]
fn pad_fill_is_exactly_gray_127() {
    // 50x50 into a 28x14 target: `contain` yields 14x14, so columns 0..7 and 21..28 are fill.
    let src = gradient_rgb(50, 50);
    let out = pad_contain_gray(&src, 50, 50, 28, 14);
    assert_eq!(out.len(), 28 * 14 * 3);
    let px = |x: usize, y: usize| {
        let i = (y * 28 + x) * 3;
        [out[i], out[i + 1], out[i + 2]]
    };
    for y in 0..14 {
        assert_eq!(px(0, y), [127, 127, 127], "left fill column at row {y}");
        assert_eq!(px(27, y), [127, 127, 127], "right fill column at row {y}");
    }
    assert_ne!(
        px(14, 7),
        [127, 127, 127],
        "the centre must hold image, not fill"
    );
}

/// Python's `round()` is banker's rounding, and `ImageOps.contain`/`pad` depend on it for
/// the contained dimension and the paste offset. Half-away-from-zero is a real, silent
/// one-pixel shift.
#[test]
fn py_round_is_half_to_even() {
    let f = fixture();
    let table = rows(&f, "py_round_half_even");
    assert!(table.len() >= 10);
    for row in table {
        let pair = row.as_array().expect("pair");
        let x = pair[0].as_f64().expect("f64");
        let want = pair[1].as_i64().expect("i64");
        assert_eq!(py_round_half_even(x), want, "py_round_half_even({x})");
    }
    // Explicit, so the intent survives a fixture regeneration.
    assert_eq!(py_round_half_even(0.5), 0);
    assert_eq!(py_round_half_even(1.5), 2);
    assert_eq!(py_round_half_even(2.5), 2);
    assert_eq!(py_round_half_even(-0.5), 0);
    assert_eq!(py_round_half_even(-1.5), -2);
}
