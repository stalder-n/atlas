// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the end-to-end image preprocessing: decode → plan → resample →
//! patchify, the pixel half of the reference's `load_image`.
//!
//! The geometry expectations come from `fixtures/geometry.json` (the reference's own
//! numbers); the pixel layout is checked against the resampler this module composes, plus
//! one hand-computed patch element that pins the `(c, y, x)` inner order.

use super::*;
use crate::vision_deepseek::geometry::{DsImagePlan, DsVisionGeometry, plan_image};
use crate::vision_deepseek::resample::{pad_contain_gray, resize_pil_bicubic_rgb};
use crate::vision_deepseek::testutil::{gradient_png_data_uri, gradient_rgb};
use serde_json::Value;

const GEOMETRY_FIXTURE: &str = include_str!("fixtures/geometry.json");

fn geometry() -> DsVisionGeometry {
    let f: Value = serde_json::from_str(GEOMETRY_FIXTURE).expect("geometry.json parses");
    let g = &f["geometry"];
    let u = |k: &str| g[k].as_u64().unwrap_or_else(|| panic!("missing {k}")) as usize;
    DsVisionGeometry::new(
        u("patch_size"),
        u("downsample_ratio"),
        u("max_n_token"),
        u("min_pixels"),
        u("max_wh_ratio"),
    )
    .expect("the checkpoint's geometry")
}

/// A synthetic PNG round-trips to `n_vit_h · n_vit_w · 3 · p · p` (= ·588 at p = 14) floats
/// in [-1, 1], on the grid the reference planned.
#[test]
fn png_round_trip_yields_the_planned_patch_grid() {
    let g = geometry();
    let p = g.patch_size;
    for (w, h) in [
        (1usize, 1usize),
        (64, 64),
        (200, 150),
        (97, 61),
        (1024, 768),
    ] {
        let uri = gradient_png_data_uri(w, h);
        let out = preprocess_image_deepseek(&uri, &g)
            .unwrap_or_else(|e| panic!("preprocess {w}x{h}: {e:#}"));
        let want_plan = plan_image(w, h, &g).expect("plan");
        assert_eq!(
            out.plan, want_plan,
            "{w}x{h}: plan must come from plan_image"
        );
        assert_eq!(
            out.pixels.len(),
            out.plan.n_vit_h * out.plan.n_vit_w * 3 * p * p,
            "{w}x{h}: patch buffer length"
        );
        assert_eq!(
            3 * p * p,
            588,
            "the DeepSeek ViT patch is 3*14*14 = 588 floats"
        );
        for (i, v) in out.pixels.iter().enumerate() {
            assert!(
                v.is_finite() && (-1.0..=1.0).contains(v),
                "{w}x{h}: pixel {i} = {v} outside [-1, 1]"
            );
        }
    }
}

/// A plan for a grid that is already an exact multiple of the patch size.
fn exact_grid_plan(best_width: usize, best_height: usize, p: usize) -> DsImagePlan {
    DsImagePlan {
        best_width,
        best_height,
        n_vit_h: best_height / p,
        n_vit_w: best_width / p,
        n_llm_h: best_height / p / 3,
        n_llm_w: best_width / p / 3,
        squash: false,
    }
}

/// The normalisation is the reference's `(x/255 - 0.5) / 0.5`, and the layout is
/// patch-major with `(c, y, x)` inner — the reference's
/// `reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4)`.
#[test]
fn patch_layout_is_patch_major_with_channel_y_x_inner() {
    let g = geometry();
    let p = g.patch_size;
    // A grid exactly 2x3 patches, so the reshape is checkable by hand.
    let (bw, bh) = (3 * p, 2 * p);
    let plan = exact_grid_plan(bw, bh, p);
    let rgb = gradient_rgb(bw, bh);
    let pixels = patches_from_rgb8(&rgb, &plan, p);
    assert_eq!(pixels.len(), 2 * 3 * 3 * p * p);

    let norm = |b: u8| (b as f32 / 255.0 - 0.5) / 0.5;
    for (ph, pw, c, y, x) in [
        (0usize, 0usize, 0usize, 0usize, 0usize),
        (0, 0, 1, 3, 5),
        (1, 2, 2, 13, 13),
        (1, 0, 0, 7, 2),
        (0, 2, 1, 0, 13),
    ] {
        let src = ((ph * p + y) * bw + (pw * p + x)) * 3 + c;
        let dst = ((ph * 3 + pw) * 3 + c) * p * p + y * p + x;
        assert_eq!(
            pixels[dst],
            norm(rgb[src]),
            "patch ({ph},{pw}) channel {c} at ({y},{x})"
        );
    }
    // Endpoints of the normalisation.
    assert_eq!(norm(0), -1.0);
    assert_eq!(norm(255), 1.0);
}

/// `w >= max_wh_ratio * h` takes the reference's plain-resize branch (no gray bars);
/// everything else takes `ImageOps.pad`. Getting this backwards is invisible in the shapes.
#[test]
fn squash_and_pad_branches_match_the_resampler_they_name() {
    let g = geometry();
    let p = g.patch_size;
    let norm = |b: u8| (b as f32 / 255.0 - 0.5) / 0.5;

    for (w, h) in [(2000usize, 100usize), (200usize, 150usize)] {
        let plan = plan_image(w, h, &g).expect("plan");
        let src = gradient_rgb(w, h);
        let expected_rgb = if plan.squash {
            resize_pil_bicubic_rgb(&src, w, h, plan.best_width, plan.best_height)
        } else {
            pad_contain_gray(&src, w, h, plan.best_width, plan.best_height)
        };
        let out = preprocess_image_deepseek(&gradient_png_data_uri(w, h), &g).expect("preprocess");
        let want = patches_from_rgb8(&expected_rgb, &plan, p);
        assert_eq!(out.pixels.len(), want.len(), "{w}x{h}: length");
        let worst = out
            .pixels
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= norm(1) - norm(0) + 1e-6,
            "{w}x{h} (squash={}): the chosen resampler differs by {worst}",
            plan.squash
        );
    }
    assert!(
        plan_image(2000, 100, &g).expect("plan").squash,
        "2000x100 must squash"
    );
    assert!(
        !plan_image(200, 150, &g).expect("plan").squash,
        "200x150 must pad"
    );
}

/// SBIO: this function performs no I/O. A URI the server has not already resolved is an
/// error, never a fetch.
#[test]
fn non_data_uris_are_rejected_without_io() {
    let g = geometry();
    for uri in [
        "http://example.invalid/cat.png",
        "https://example.invalid/cat.png",
        "file:///etc/passwd",
        "/etc/passwd",
        "",
    ] {
        assert!(
            preprocess_image_deepseek(uri, &g).is_err(),
            "`{uri}` must be rejected: preprocess performs no I/O"
        );
    }
}

#[test]
fn malformed_payloads_error_rather_than_panic() {
    let g = geometry();
    for uri in [
        "data:image/png;base64,",
        "data:image/png;base64,!!!!not-base64!!!!",
        "data:image/png,notbase64encoded",
        "data:image/png;base64,aGVsbG8gd29ybGQ=", // valid base64, not an image
    ] {
        assert!(
            preprocess_image_deepseek(uri, &g).is_err(),
            "`{uri}` must error"
        );
    }
    // A truncated PNG: valid base64, correct magic, truncated body.
    let full = gradient_png_data_uri(32, 32);
    let comma = full.find(',').expect("data uri comma") + 1;
    let truncated = format!("{}{}", &full[..comma], &full[comma..comma + 40]);
    assert!(
        preprocess_image_deepseek(&truncated, &g).is_err(),
        "a truncated PNG must error"
    );
}

/// A 1x1 image is upscaled past `min_pixels` rather than producing an empty grid.
#[test]
fn a_one_pixel_image_produces_a_real_grid() {
    let g = geometry();
    let out = preprocess_image_deepseek(&gradient_png_data_uri(1, 1), &g).expect("1x1 preprocess");
    assert!(out.plan.n_vit_h > 0 && out.plan.n_vit_w > 0);
    assert_eq!(
        out.plan.n_vit_h, 28,
        "1x1 upscales to the min_pixels square"
    );
    assert_eq!(out.plan.n_vit_w, 28);
    assert_eq!(out.pixels.len(), 28 * 28 * 588);
}
