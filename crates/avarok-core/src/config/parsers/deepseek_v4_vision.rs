// SPDX-License-Identifier: AGPL-3.0-only

//! Vision half of the DeepSeek-V4 config parser.
//!
//! DeepSeek-V4-Flash-Vision-Exp keeps `model_type = "deepseek_v4"` and ships
//! the tower as FLAT `vision_*` keys (no nested `vision_config` block), so
//! `parsers::vision` never sees it. `vision_n_layers > 0` is exactly the
//! reference's own `vl` flag; when it is set, every other `vision_*` key is
//! REQUIRED (PCND — a zero here would silently size a buffer to nothing).

use anyhow::{Result, bail};
use serde_json::Value;

use super::super::{VisionConfig, VisionFlavor};

/// The flat keys that must accompany a declared `vision_n_layers`.
///
/// Also the exact set used to detect a half-populated config: a checkpoint
/// carrying one of these without `vision_n_layers` is malformed, not text.
const REQUIRED_VISION_KEYS: [&str; 9] = [
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

/// Build the tower config from the flat `vision_*` keys of a DeepSeek-V4
/// `config.json`.
///
/// `hidden_size` is the already-parsed LLM width; it becomes the aligner's
/// output width (`out_hidden_size`), so the number has a single source.
///
/// Returns `Ok(None)` for a text checkpoint (`vision_n_layers` absent or 0
/// and no stray vision keys). Errors instead of guessing whenever the
/// checkpoint declares a tower it does not fully describe.
pub(super) fn parse(raw: &Value, hidden_size: usize) -> Result<Option<VisionConfig>> {
    // `parse_deepseek_v4` has already sanitized every top-level `null` to 0,
    // so "absent", "null" and "0" are indistinguishable here — and all three
    // mean the same thing for every one of these keys.
    let depth = raw
        .get("vision_n_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    if depth == 0 {
        // Text checkpoint — but only if it is consistently one. A config that
        // describes half a tower must not serve silently as text.
        if let Some(stray) = REQUIRED_VISION_KEYS
            .into_iter()
            .find(|key| raw.get(*key).is_some())
        {
            bail!(
                "DeepSeek-V4 config: `{stray}` is present but `vision_n_layers` is \
                 missing or zero, so no vision tower is declared; remove `{stray}` \
                 for a text checkpoint or set `vision_n_layers`"
            );
        }
        return Ok(None);
    }

    if hidden_size == 0 {
        bail!(
            "DeepSeek-V4 vision tower requires a non-zero top-level `hidden_size` \
             for the aligner output width, but `vision_n_layers` = {depth}"
        );
    }

    Ok(Some(VisionConfig {
        depth,
        hidden_size: req_usize(raw, "vision_dim", depth)?,
        num_heads: req_usize(raw, "vision_n_heads", depth)?,
        patch_size: req_usize(raw, "vision_patch_size", depth)?,
        // Still images only: this tower has no temporal patch axis (Qwen
        // uses 2 here and pairs frames).
        temporal_patch_size: 1,
        // The 3x3 pixel-unshuffle the aligner performs before its Linear.
        spatial_merge_size: req_usize(raw, "vision_downsample_ratio", depth)?,
        intermediate_size: req_usize(raw, "vision_inter_dim", depth)?,
        // Aligner output width, taken from the already-parsed LLM hidden size
        // rather than re-read from `raw` (SSOT).
        out_hidden_size: hidden_size,
        // DeepSeek's tower feeds the LLM once, at the aligner; there are no
        // intermediate deepstack taps.
        deepstack_visual_indexes: Vec::new(),
        // The placeholder token id is NOT in config.json: the DeepSeek
        // placeholder is resolved from `tokenizer.json` by the server. 0 is
        // the documented contract value, not a PCND default.
        image_pad_token_id: 0,
        // No video path for this tower.
        video_pad_token_id: 0,
        // Resolved at serve time by `resolve_vision_max_pixels`, same as Qwen.
        max_pixels: None,
        flavor: VisionFlavor::DeepseekV4,
        rope_theta: req_f64(raw, "vision_rope_theta", depth)?,
        max_n_token: req_usize(raw, "vision_max_n_token", depth)?,
        min_pixels: req_usize(raw, "vision_min_pixels", depth)?,
        max_wh_ratio: req_usize(raw, "vision_max_wh_ratio", depth)?,
    }))
}

/// Read a required positive integer key, treating missing and zero alike.
fn req_usize(raw: &Value, key: &str, depth: usize) -> Result<usize> {
    match raw.get(key).and_then(Value::as_u64) {
        Some(v) if v > 0 => Ok(v as usize),
        _ => bail!(
            "DeepSeek-V4 vision config: `{key}` is missing or zero, but \
             `vision_n_layers` = {depth} declares a vision tower"
        ),
    }
}

/// Read a required positive float key, treating missing and zero alike.
fn req_f64(raw: &Value, key: &str, depth: usize) -> Result<f64> {
    match raw.get(key).and_then(Value::as_f64) {
        Some(v) if v > 0.0 => Ok(v),
        _ => bail!(
            "DeepSeek-V4 vision config: `{key}` is missing or zero, but \
             `vision_n_layers` = {depth} declares a vision tower"
        ),
    }
}

#[cfg(test)]
#[path = "deepseek_v4_vision_tests.rs"]
mod deepseek_v4_vision_tests;
