// SPDX-License-Identifier: AGPL-3.0-only

//! Parser tests pinned to the real `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
//! `config.json` (fetched 2026-09-21, byte-verbatim in `fixtures/`).
//!
//! Everything goes through the PUBLIC [`parse_config`], so what is pinned is
//! the whole dispatch → `parse_deepseek_v4` → vision-helper chain, not a
//! helper in isolation: a future refactor that moves the flat-key handling
//! elsewhere still has to satisfy these.

use serde_json::Value;

use crate::config::{ModelConfig, VisionFlavor, parse_config};

/// The upstream Vision-Exp config, verbatim.
const VISION_EXP: &str = include_str!("fixtures/vision-exp-config.json");

fn fixture_json() -> Value {
    serde_json::from_str(VISION_EXP).expect("Vision-Exp fixture is valid JSON")
}

/// Re-serialize a mutated fixture and hand it to the public entry point.
fn parse_mutated(
    mutate: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> anyhow::Result<ModelConfig> {
    let mut raw = fixture_json();
    mutate(raw.as_object_mut().expect("fixture is a JSON object"));
    parse_config(&raw.to_string())
}

#[test]
fn vision_exp_fixture_populates_every_c1_field() {
    let cfg = parse_config(VISION_EXP).expect("Vision-Exp config parses");
    let v = cfg
        .vision
        .as_ref()
        .expect("`vision_n_layers` = 32 declares a tower, so `vision` must be Some");

    assert_eq!(v.flavor, VisionFlavor::DeepseekV4, "flavor");
    assert_eq!(v.depth, 32, "depth <- vision_n_layers");
    assert_eq!(v.hidden_size, 1024, "hidden_size <- vision_dim");
    assert_eq!(v.num_heads, 16, "num_heads <- vision_n_heads");
    assert_eq!(v.patch_size, 14, "patch_size <- vision_patch_size");
    assert_eq!(v.temporal_patch_size, 1, "DS tower has no temporal axis");
    assert_eq!(
        v.spatial_merge_size, 3,
        "spatial_merge_size <- vision_downsample_ratio"
    );
    assert_eq!(
        v.intermediate_size, 2816,
        "intermediate_size <- vision_inter_dim"
    );
    assert_eq!(
        v.out_hidden_size, 4096,
        "out_hidden_size is the LLM hidden_size, the aligner's output width"
    );
    assert_eq!(
        v.out_hidden_size, cfg.hidden_size,
        "aligner output width is SSOT-derived"
    );
    assert!(v.deepstack_visual_indexes.is_empty(), "DS has no deepstack");
    assert_eq!(
        v.image_pad_token_id, 0,
        "placeholder id comes from the tokenizer"
    );
    assert_eq!(v.video_pad_token_id, 0, "no video path");
    assert_eq!(
        v.max_pixels, None,
        "resolved at serve time, not from config.json"
    );
    assert_eq!(v.rope_theta, 10000.0, "rope_theta <- vision_rope_theta");
    assert_eq!(v.max_n_token, 384, "max_n_token <- vision_max_n_token");
    assert_eq!(v.min_pixels, 147_456, "min_pixels <- vision_min_pixels");
    assert_eq!(v.max_wh_ratio, 8, "max_wh_ratio <- vision_max_wh_ratio");

    // Consistency pins: the aligner is Linear(9216 -> 4096) and the ViT head
    // dim is 64. Both are derived, never configured.
    assert_eq!(
        v.merger_input_size(),
        9216,
        "3*3*1024 pixel-unshuffle input"
    );
    assert_eq!(v.hidden_size / v.num_heads, 64, "ViT head_dim");
}

#[test]
fn vision_exp_rms_norm_eps_reaches_the_config_unclamped() {
    let cfg = parse_config(VISION_EXP).expect("Vision-Exp config parses");
    assert_eq!(
        cfg.rms_norm_eps, 1e-20,
        "DeepSeek-V4 ships 1e-20; any clamp to a 'sane' epsilon changes every norm"
    );
}

#[test]
fn vision_exp_num_nextn_predict_layers_becomes_three_mtp_modules() {
    let cfg = parse_config(VISION_EXP).expect("Vision-Exp config parses");
    assert_eq!(cfg.num_mtp_modules, 3, "num_nextn_predict_layers = 3");
    assert_eq!(cfg.mtp_num_hidden_layers, 3);
    assert_eq!(cfg.mtp_transformer_layers, 1);
}

#[test]
fn vision_exp_pins_the_sentinel_and_window_sources() {
    let cfg = parse_config(VISION_EXP).expect("Vision-Exp config parses");
    // These two are the PCND sources downstream units read for the sentinel
    // id floor and the image-span attention window. Pinning them here is what
    // makes "no literals in production code" checkable.
    assert_eq!(
        cfg.vocab_size, 129_280,
        "sentinel ids live at >= vocab_size"
    );
    assert_eq!(cfg.sliding_window, 128, "the attention window SSOT");
}

#[test]
fn text_checkpoint_has_no_vision_tower() {
    // A 0731-shaped text config, derived from the one owned fixture by
    // stripping every `vision_*` key. Deriving rather than shipping a second
    // fixture proves the `vision.is_some() <=> vision_n_layers > 0`
    // equivalence directly, on otherwise byte-identical input.
    let text = parse_mutated(|obj| {
        obj.retain(|k, _| !k.starts_with("vision_"));
    })
    .expect("text checkpoint parses");
    assert!(text.vision.is_none(), "no vision_* keys => no tower");

    // ...and nothing else moved: text checkpoints must be unaffected.
    let vl = parse_config(VISION_EXP).expect("Vision-Exp config parses");
    assert_eq!(text.rms_norm_eps, vl.rms_norm_eps);
    assert_eq!(text.rope_theta, vl.rope_theta);
    assert_eq!(text.head_dim, vl.head_dim);
    assert_eq!(text.vocab_size, vl.vocab_size);
    assert_eq!(text.sliding_window, vl.sliding_window);
    assert_eq!(text.num_mtp_modules, vl.num_mtp_modules);
    assert_eq!(text.dspark_block_size, vl.dspark_block_size);
    assert_eq!(text.dspark_noise_token_id, vl.dspark_noise_token_id);
    assert_eq!(text.dspark_target_layer_ids, vl.dspark_target_layer_ids);
    assert_eq!(text.dspark_markov_rank, vl.dspark_markov_rank);
}

#[test]
fn each_vision_key_is_required_when_the_tower_is_declared() {
    // Written out here independently of production on purpose: this list is
    // WHICH keys a declared tower must carry, not "whatever
    // `REQUIRED_VISION_KEYS` happens to contain today". Iterating the
    // production const instead would make the pin circular — dropping a key
    // from it and defaulting the field (a PCND violation) would still pass.
    const REQUIRED: [&str; 9] = [
        "vision_dim",
        "vision_n_heads",
        "vision_inter_dim",
        "vision_patch_size",
        "vision_rope_theta",
        "vision_downsample_ratio",
        "vision_max_n_token",
        "vision_min_pixels",
        "vision_max_wh_ratio",
    ];

    for missing in REQUIRED {
        let err = parse_mutated(|obj| {
            obj.remove(missing);
        })
        .expect_err("a declared tower with a missing key must fail fast");
        assert!(
            err.to_string().contains(missing),
            "removing `{missing}` produced an error that does not name it: {err:#}"
        );
    }

    // The two lists pin each other: a key added to production without a case
    // here (or removed from here without removing the requirement) is caught.
    // Compared as slices so a length change is a test failure, not a compile
    // error — the loop above must be the one that reports the real regression.
    assert_eq!(
        REQUIRED.as_slice(),
        super::REQUIRED_VISION_KEYS.as_slice(),
        "production's required-key set drifted from this test's list"
    );
}

#[test]
fn a_zero_vision_key_is_rejected_like_a_missing_one() {
    // Zero is indistinguishable from missing here: `parse_deepseek_v4`
    // sanitizes every top-level `null` to 0 before deserializing, and 0 is
    // meaningless for all nine keys.
    let err = parse_mutated(|obj| {
        obj.insert("vision_dim".into(), Value::from(0));
    })
    .expect_err("`vision_dim: 0` must fail fast");
    assert!(
        err.to_string().contains("vision_dim"),
        "zero `vision_dim` error does not name the key: {err:#}"
    );

    let err = parse_mutated(|obj| {
        obj.insert("vision_rope_theta".into(), Value::Null);
    })
    .expect_err("`vision_rope_theta: null` must fail fast (null -> 0)");
    assert!(
        err.to_string().contains("vision_rope_theta"),
        "null `vision_rope_theta` error does not name the key: {err:#}"
    );
}

#[test]
fn a_stray_vision_key_without_vision_n_layers_is_rejected() {
    // A half-populated config must not serve silently as text.
    let err = parse_mutated(|obj| {
        obj.retain(|k, _| !k.starts_with("vision_") || k == "vision_dim");
    })
    .expect_err("`vision_dim` without `vision_n_layers` must fail fast");
    assert!(
        err.to_string().contains("vision_dim"),
        "stray-key error does not name the offender: {err:#}"
    );
}

#[test]
fn the_qwen_parser_keeps_the_qwen_flavor() {
    // Guards against a future edit flipping the shared struct's defaults: the
    // nested-block path must stay Qwen-flavored with sentinel DS numerics.
    let raw: Value = serde_json::from_str(
        r#"{"vision_config":{"depth":27,"hidden_size":1152,"num_heads":16,
            "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
            "intermediate_size":4304,"out_hidden_size":2048}}"#,
    )
    .expect("valid JSON");
    let v = super::super::vision::parse_vision_config(&raw).expect("nested block parses");
    assert_eq!(v.flavor, VisionFlavor::Qwen);
    assert_eq!(v.rope_theta, 10000.0, "Qwen family RoPE base");
    assert_eq!(v.max_n_token, 0, "DS-only sentinel");
    assert_eq!(v.min_pixels, 0, "DS-only sentinel");
    assert_eq!(v.max_wh_ratio, 0, "DS-only sentinel");
}
