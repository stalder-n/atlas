// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the DeepSeek-V4 image SIZING port.
//!
//! Every expectation is a number DeepSeek's own `inference/image_processor.py` returned at
//! revision `86f746b3…`, recorded by `gen_layout_goldens.py` into `fixtures/geometry.json`.
//! Nothing here is hand-derived, so a test that passes proves reference parity rather than
//! self-consistency.

use super::*;
use crate::vision_deepseek::block;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/geometry.json");
const REVISION: &str = "86f746b36186f0e567729a5c06a8c918caba82a9";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/geometry.json parses")
}

fn u(v: &Value, k: &str) -> usize {
    v.get(k)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("golden row is missing the integer field `{k}`: {v}"))
        as usize
}

fn b(v: &Value, k: &str) -> bool {
    v.get(k)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("golden row is missing the bool field `{k}`: {v}"))
}

fn rows<'a>(f: &'a Value, k: &str) -> &'a Vec<Value> {
    f[k].as_array()
        .unwrap_or_else(|| panic!("fixture has no `{k}` array"))
}

fn geometry_of(f: &Value) -> DsVisionGeometry {
    let g = &f["geometry"];
    DsVisionGeometry::new(
        u(g, "patch_size"),
        u(g, "downsample_ratio"),
        u(g, "max_n_token"),
        u(g, "min_pixels"),
        u(g, "max_wh_ratio"),
    )
    .expect("the checkpoint's own geometry must be accepted")
}

/// Pins the fixture itself: right checkpoint, right revision, and the coverage the spec
/// demands. A fixture swapped for a weaker one fails here rather than silently passing
/// every comparison below. (This is the one test in the file that does not call production
/// code — it guards the goldens, not the port.)
#[test]
fn golden_provenance_and_coverage() {
    let f = fixture();
    assert_eq!(f["reference"]["revision"], REVISION);
    assert_eq!(
        f["reference"]["repo"], "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "goldens must come from the vision checkpoint, not the text one"
    );
    assert!(
        f["reference"]["files"]["image_processor.py"].is_string(),
        "the header must carry a sha256 per reference file"
    );
    let g = &f["geometry"];
    assert_eq!(
        (
            u(g, "patch_size"),
            u(g, "downsample_ratio"),
            u(g, "max_n_token")
        ),
        (14, 3, 384)
    );
    assert_eq!((u(g, "min_pixels"), u(g, "max_wh_ratio")), (147_456, 8));

    let plan = rows(&f, "plan");
    assert!(
        plan.len() >= 40,
        "spec requires >= 40 plan cases, got {}",
        plan.len()
    );
    assert!(plan.iter().any(|r| b(r, "squash")), "no squash case");
    assert!(plan.iter().any(|r| !b(r, "squash")), "no pad case");
    let budget = u(g, "max_n_token") - (COMPRESS_PAD_TO - 1);
    let looped = plan
        .iter()
        .filter(|r| u(r, "initial_num_tokens") > budget)
        .count();
    assert!(
        looped >= 8,
        "only {looped} cases iterate the safe_resize budget loop"
    );

    let branches: std::collections::BTreeSet<&str> = rows(&f, "solve")
        .iter()
        .map(|r| r["branch"].as_str().expect("branch"))
        .collect();
    assert_eq!(
        branches,
        ["else", "max_h_float<2", "max_w_float<1"]
            .into_iter()
            .collect(),
        "all three solve_resize_ratio branches must be covered"
    );
    assert!(
        rows(&f, "solve").iter().any(|r| b(r, "assert_fails")),
        "no case reaches the reference's `assert max_w > 1`"
    );
}

/// `plan_image` reproduces the reference's whole sizing pipeline — the `max_wh_ratio`
/// clamp, the `min_pixels` upscale, the patch-ceiling, `safe_resize`, and the squash
/// decision taken from the ORIGINAL dimensions — for every golden size.
#[test]
fn plan_image_matches_the_reference_for_every_golden_size() {
    let f = fixture();
    let g = geometry_of(&f);
    for row in rows(&f, "plan") {
        let (w, h) = (u(row, "w"), u(row, "h"));
        let note = row["note"].as_str().unwrap_or("");
        let got = plan_image(w, h, &g)
            .unwrap_or_else(|e| panic!("plan_image({w}x{h}) [{note}] failed: {e:#}"));
        let want = DsImagePlan {
            best_width: u(row, "best_w"),
            best_height: u(row, "best_h"),
            n_vit_h: u(row, "n_vit_h"),
            n_vit_w: u(row, "n_vit_w"),
            n_llm_h: u(row, "n_llm_h"),
            n_llm_w: u(row, "n_llm_w"),
            squash: b(row, "squash"),
        };
        assert_eq!(got, want, "plan_image({w}x{h}) [{note}]");
    }
}

/// The `(width, height)` argument order of `plan_image` is the inverse of every reference
/// helper's `(height, width)`. A swapped wrapper passes on squares and fails here.
#[test]
fn plan_image_argument_order_is_width_then_height() {
    let f = fixture();
    let g = geometry_of(&f);
    let row = rows(&f, "plan")
        .iter()
        .find(|r| u(r, "w") == 100 && u(r, "h") == 1000)
        .expect("the 100x1000 tall case must be in the golden");
    let got = plan_image(100, 1000, &g).expect("tall image plans");
    assert_eq!(got.n_llm_h, u(row, "n_llm_h"));
    assert_eq!(got.n_llm_w, u(row, "n_llm_w"));
    assert!(
        got.n_llm_h > got.n_llm_w,
        "a 100x1000 (tall) image must yield more rows than columns, got {got:?}"
    );
}

#[test]
fn grid_tokens_matches_the_reference() {
    let f = fixture();
    let all = rows(&f, "grid_tokens");
    assert!(all.len() >= 200, "thin grid_tokens sweep: {}", all.len());
    for row in all {
        let (bh, bw, p, ds) = (
            u(row, "best_h"),
            u(row, "best_w"),
            u(row, "patch"),
            u(row, "ds"),
        );
        assert_eq!(
            grid_tokens(bh, bw, p, ds),
            (u(row, "n_llm_h"), u(row, "n_llm_w"), u(row, "num_tokens")),
            "grid_tokens({bh}, {bw}, {p}, {ds})"
        );
    }
}

#[test]
fn solve_resize_ratio_matches_the_reference_on_all_three_branches() {
    let f = fixture();
    for row in rows(&f, "solve") {
        if b(row, "assert_fails") {
            continue;
        }
        let (h, w, p, ds, budget) = (
            u(row, "h"),
            u(row, "w"),
            u(row, "patch"),
            u(row, "ds"),
            u(row, "max_n_token"),
        );
        let branch = row["branch"].as_str().unwrap_or("");
        let got = solve_resize_ratio(h, w, p, ds, budget).unwrap_or_else(|e| {
            panic!("solve_resize_ratio({h},{w},..,{budget}) [{branch}]: {e:#}")
        });
        assert_eq!(
            got,
            (
                u(row, "n_llm_h"),
                u(row, "n_llm_w"),
                u(row, "best_h"),
                u(row, "best_w"),
                u(row, "num_tokens")
            ),
            "solve_resize_ratio({h},{w},{p},{ds},{budget}) [{branch}]"
        );
    }
}

/// The reference's `assert max_w > 1` in the `max_h_float < 2` branch becomes an `Err`,
/// never a panic and never a silently degenerate grid.
#[test]
fn solve_resize_ratio_errors_where_the_reference_asserts() {
    let f = fixture();
    let mut checked = 0usize;
    for row in rows(&f, "solve") {
        if !b(row, "assert_fails") {
            continue;
        }
        let (h, w, p, ds, budget) = (
            u(row, "h"),
            u(row, "w"),
            u(row, "patch"),
            u(row, "ds"),
            u(row, "max_n_token"),
        );
        assert!(
            solve_resize_ratio(h, w, p, ds, budget).is_err(),
            "solve_resize_ratio({h},{w},{p},{ds},{budget}) must Err where the reference asserts"
        );
        checked += 1;
    }
    assert!(checked >= 2, "golden lost its assert-failing cases");
}

/// `safe_resize` driven directly on the reference's own prologue output, so the budget loop
/// is exercised independently of `plan_image`.
#[test]
fn safe_resize_matches_the_reference() {
    let f = fixture();
    let g = geometry_of(&f);
    let mut looped = 0usize;
    for row in rows(&f, "plan") {
        let got = safe_resize(
            u(row, "pre_h"),
            u(row, "pre_w"),
            u(row, "pre_best_h"),
            u(row, "pre_best_w"),
            g.patch_size,
            g.downsample_ratio,
            g.max_n_token,
        )
        .unwrap_or_else(|e| panic!("safe_resize for {}x{}: {e:#}", u(row, "w"), u(row, "h")));
        assert_eq!(
            got,
            (
                u(row, "n_llm_h"),
                u(row, "n_llm_w"),
                u(row, "best_h"),
                u(row, "best_w")
            ),
            "safe_resize for source {}x{}",
            u(row, "w"),
            u(row, "h")
        );
        if u(row, "initial_num_tokens") > g.max_n_token - (COMPRESS_PAD_TO - 1) {
            looped += 1;
        }
    }
    assert!(looped >= 8, "the budget loop was never exercised");
}

/// The reference's `while` loop has no termination proof and this is host code reachable
/// from an unauthenticated HTTP body: non-convergence must be a bounded `Err`, not a hang
/// and not a panic.
#[test]
fn safe_resize_bails_instead_of_spinning() {
    let f = fixture();
    let g = geometry_of(&f);
    // A budget so small that no grid can ever fit inside it.
    let r = safe_resize(4096, 4096, 4096, 4096, g.patch_size, g.downsample_ratio, 8);
    assert!(
        r.is_err(),
        "safe_resize must give up on an unsatisfiable budget, got {r:?}"
    );
}

#[test]
fn n_llm_is_ceiling_division() {
    for (n_vit, ds, want) in [
        (0usize, 3usize, 0usize),
        (1, 3, 1),
        (2, 3, 1),
        (3, 3, 1),
        (4, 3, 2),
        (9, 3, 3),
        (10, 3, 4),
        (28, 3, 10),
        (47, 3, 16),
        (564, 3, 188),
        (7, 1, 7),
        (7, 2, 4),
    ] {
        assert_eq!(n_llm(n_vit, ds), want, "n_llm({n_vit}, {ds})");
    }
    // Consistency with the goldens: n_llm_* is exactly n_llm(n_vit_*, downsample_ratio).
    let f = fixture();
    let ds = u(&f["geometry"], "downsample_ratio");
    for row in rows(&f, "plan") {
        assert_eq!(n_llm(u(row, "n_vit_h"), ds), u(row, "n_llm_h"));
        assert_eq!(n_llm(u(row, "n_vit_w"), ds), u(row, "n_llm_w"));
    }
}

/// The invariant the whole design rests on: every real image's expanded block fits inside
/// `vision_max_n_token` at every compressor phase, and the block length is exactly the
/// grid's token count plus the leading compress pad.
#[test]
fn every_golden_plan_fits_the_token_budget_at_every_start_phase() {
    let f = fixture();
    let g = geometry_of(&f);
    for row in rows(&f, "plan") {
        let (nh, nw, num_tokens) = (u(row, "n_llm_h"), u(row, "n_llm_w"), u(row, "num_tokens"));
        for start_pos in 0..8usize {
            let len = block::block_len(nh, nw, start_pos);
            let compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO;
            assert_eq!(
                len,
                num_tokens + compress_pad,
                "block_len({nh},{nw},{start_pos}) vs grid_tokens for {}x{}",
                u(row, "w"),
                u(row, "h")
            );
            assert!(
                len <= g.max_n_token,
                "block for {}x{} at start_pos {start_pos} is {len} > max_n_token {}",
                u(row, "w"),
                u(row, "h"),
                g.max_n_token
            );
        }
    }
}

#[test]
fn geometry_rejects_degenerate_values() {
    assert!(
        DsVisionGeometry::new(0, 3, 384, 147_456, 8).is_err(),
        "patch_size 0"
    );
    assert!(
        DsVisionGeometry::new(14, 0, 384, 147_456, 8).is_err(),
        "downsample_ratio 0"
    );
    assert!(
        DsVisionGeometry::new(14, 3, 0, 147_456, 8).is_err(),
        "max_n_token 0"
    );
    assert!(
        DsVisionGeometry::new(14, 3, 15, 147_456, 8).is_err(),
        "max_n_token below the floor"
    );
    // `min_pixels` / `max_wh_ratio` of 0 are the documented "no clamp / no upscale"
    // sentinels, not errors.
    assert!(DsVisionGeometry::new(14, 3, 384, 0, 0).is_ok());
}

#[test]
fn plan_image_rejects_a_zero_dimension() {
    let f = fixture();
    let g = geometry_of(&f);
    assert!(plan_image(0, 100, &g).is_err());
    assert!(plan_image(100, 0, &g).is_err());
    assert!(plan_image(0, 0, &g).is_err());
}

/// `from_vision_config` is the C1 adapter: DeepSeek flavour only, values taken from the
/// config SSOT rather than from literals.
#[test]
fn from_vision_config_takes_deepseek_and_rejects_qwen() {
    use avarok_core::config::{VisionConfig, VisionFlavor};

    let base = VisionConfig {
        depth: 32,
        hidden_size: 1024,
        num_heads: 16,
        patch_size: 14,
        temporal_patch_size: 1,
        spatial_merge_size: 3,
        intermediate_size: 2816,
        out_hidden_size: 4096,
        deepstack_visual_indexes: vec![],
        image_pad_token_id: 0,
        video_pad_token_id: 0,
        max_pixels: None,
        flavor: VisionFlavor::DeepseekV4,
        rope_theta: 10_000.0,
        max_n_token: 384,
        min_pixels: 147_456,
        max_wh_ratio: 8,
    };
    let g = DsVisionGeometry::from_vision_config(&base).expect("DeepSeek flavour is accepted");
    assert_eq!(
        g,
        DsVisionGeometry::new(14, 3, 384, 147_456, 8).expect("valid"),
        "geometry must be read from the config, not from literals"
    );

    let qwen = VisionConfig {
        flavor: VisionFlavor::Qwen,
        max_n_token: 0,
        min_pixels: 0,
        max_wh_ratio: 0,
        ..base.clone()
    };
    assert!(
        DsVisionGeometry::from_vision_config(&qwen).is_err(),
        "a Qwen tower must never drive the DeepSeek geometry"
    );
}
