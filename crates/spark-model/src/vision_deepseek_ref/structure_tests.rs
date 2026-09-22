// SPDX-License-Identifier: AGPL-3.0-only

//! Structural properties of the DeepSeek-V4 host ViT/aligner reference (unit U1.3).
//!
//! The golden in `tests.rs` proves reference parity; these four cases pin the properties a
//! golden cannot express — exact zero padding, the RoPE half split, the derived dimensions
//! and the loud rejection of a mis-shaped input. Split out of `tests.rs` only to keep both
//! files under the 500-LoC cap; the fixture helpers stay in `tests.rs` (the SSOT for them).

use super::tests::{dims, fixture, weights};
use super::*;

/// A 1x1 grid at ds=3 fills exactly one of nine slots per channel; the other eight are
/// EXACT zeros, not small numbers.
#[test]
fn unshuffle_pads_with_exact_zeros() {
    let (dim, ds) = (4usize, 3usize);
    let row: Vec<f32> = (0..dim).map(|c| 1.0 + c as f32).collect();
    let out = pixel_unshuffle(&row, 1, 1, dim, ds);
    assert_eq!(out.len(), dim * ds * ds);
    for c in 0..dim {
        for k in 0..ds * ds {
            let want = if k == 0 { 1.0 + c as f32 } else { 0.0 };
            assert_eq!(
                out[c * ds * ds + k],
                want,
                "slot c={c} k={k} must be {want} (index is c·ds²+kh·ds+kw)"
            );
        }
    }
}

/// Row 0 sits at `h = w = 0`, so every angle is zero; a single-column grid has zero width
/// angles in the second half of every row. Pins the half split and the row ordering.
#[test]
fn rope_zero_positions_are_identity() {
    let (cos, sin) = rope_cos_sin(3, 1, 8, 10000.0);
    assert_eq!(cos.len(), 24);
    for j in 0..8 {
        assert_eq!(cos[j], 1.0, "cos row 0 col {j}");
        assert_eq!(sin[j], 0.0, "sin row 0 col {j}");
    }
    for r in 0..3 {
        for j in 4..8 {
            assert_eq!(
                cos[r * 8 + j],
                1.0,
                "width angle must be zero at row {r} col {j}"
            );
            assert_eq!(
                sin[r * 8 + j],
                0.0,
                "width angle must be zero at row {r} col {j}"
            );
        }
    }
    assert!(sin[8] > 0.0, "row 1 carries a non-zero height angle");
}

/// Derived dimensions come from `DsVitDims`, never from a literal at a call site.
#[test]
fn derived_dims_are_exposed() {
    let d = dims(&fixture());
    assert_eq!((d.head_dim(), d.rope_dim(), d.patch_in()), (16, 8, 12));
}

/// A mis-shaped input is a loud failure, never a silent truncation.
#[test]
#[should_panic(expected = "vit_forward: expected")]
fn a_misshaped_patch_slice_is_rejected() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    let short = vec![0.0f32; 3 * d.patch_in()];
    vit_forward(&w, &d, &short, 2, 2);
}
