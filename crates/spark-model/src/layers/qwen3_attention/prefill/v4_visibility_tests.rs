// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the image-span attention bounds. No GPU: `ImageWindow` is
//! pure arithmetic plus a device-offset layout, and the pack test drives a
//! real [`PinnedPacker`] over a zeroed heap allocation the same way the
//! packer's own tests stand in for the pinned region.

use super::*;
use crate::model::pinned_pack::PinnedPacker;
use crate::vision_deepseek::block;
use avarok_core::config::VisionConfig;

const VOCAB: usize = 129_280;
const WINDOW: u32 = 128;
const MAX_N_TOKEN: usize = 384;
/// The `<image>` placeholder id the server expands; any non-sentinel id works.
const PLACEHOLDER: u32 = 128_815;

fn ds_vision_config() -> VisionConfig {
    VisionConfig {
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 1,
        spatial_merge_size: 1,
        intermediate_size: 4304,
        out_hidden_size: 2048,
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: PLACEHOLDER,
        video_pad_token_id: 0,
        max_pixels: None,
        flavor: VisionFlavor::DeepseekV4,
        rope_theta: 10_000.0,
        max_n_token: MAX_N_TOKEN,
        min_pixels: 147_456,
        max_wh_ratio: 8,
    }
}

/// A V4-shaped config: the three keys `for_chunk` reads, spelled explicitly.
fn config(vision: Option<VisionConfig>) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.vocab_size = VOCAB;
    c.sliding_window = WINDOW;
    c.vision = vision;
    c
}

/// `[text… , <image>, text…]` expanded through the SSOT block builder, so the
/// sentinel layout under test is the real one rather than a hand-rolled twin.
fn expanded(grids: &[(usize, usize)]) -> Vec<u32> {
    let mut raw: Vec<u32> = (1u32..=40).collect();
    for (i, _) in grids.iter().enumerate() {
        raw.push(PLACEHOLDER);
        raw.extend((0..7).map(|k| 200 + (i as u32) * 10 + k));
    }
    raw.extend(41u32..=90);
    let (tokens, blocks) =
        block::expand_placeholders(&raw, PLACEHOLDER, VOCAB as u32, grids).unwrap();
    assert_eq!(blocks.len(), grids.len());
    tokens
}

#[test]
fn text_only_streams_and_non_ds_configs_yield_none() {
    let text: Vec<u32> = (1u32..=200).collect();
    assert!(
        ImageWindow::for_chunk(&config(Some(ds_vision_config())), &text, 0)
            .unwrap()
            .is_none(),
        "no sentinel in the stream ⇒ no bounds"
    );
    assert!(
        ImageWindow::for_chunk(&config(None), &text, 0)
            .unwrap()
            .is_none(),
        "text checkpoint ⇒ no bounds"
    );

    // Sentinel-LOOKING ids on a Qwen-flavoured config must not be read as
    // DeepSeek sentinels: the flavor gate comes before `has_image_span`.
    let mut qwen = ds_vision_config();
    qwen.flavor = VisionFlavor::Qwen;
    let with_ids = expanded(&[(4, 6)]);
    assert!(
        ImageWindow::for_chunk(&config(Some(qwen)), &with_ids, 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_two_image_prompt_yields_bounds_the_kernel_can_take() {
    let tokens = expanded(&[(4, 6), (3, 5)]);
    let w = ImageWindow::for_chunk(&config(Some(ds_vision_config())), &tokens, 0)
        .unwrap()
        .expect("a sentinel stream on a DS vision config must produce bounds");

    assert_eq!(w.lo.len(), tokens.len());
    assert_eq!(w.hi.len(), tokens.len());
    for t in 0..tokens.len() {
        assert!(
            w.lo[t] <= w.hi[t],
            "query {t}: lo {} > hi {}",
            w.lo[t],
            w.hi[t]
        );
        assert!(
            (w.hi[t] as usize) < tokens.len(),
            "query {t}: hi {} must index the chunk ({} tokens)",
            w.hi[t],
            tokens.len()
        );
    }

    // At least one query must reach FORWARD — otherwise the stream under test
    // is not exercising the bidirectional span at all.
    assert!(
        (0..tokens.len()).any(|t| w.hi[t] as usize > t),
        "no query sees a future key: the image span is not bidirectional here"
    );
}

#[test]
fn bounds_come_from_config_not_from_literals() {
    let tokens = expanded(&[(4, 6)]);
    let cfg = config(Some(ds_vision_config()));
    let w = ImageWindow::for_chunk(&cfg, &tokens, 0).unwrap().unwrap();
    let reference = visibility::window_bounds(
        &tokens,
        VOCAB as u32,
        cfg.sliding_window as usize,
        cfg.vision.as_ref().unwrap().max_n_token,
    );
    assert_eq!(w.lo, reference.lo);
    assert_eq!(w.hi, reference.hi);

    // Change the two config knobs and the bounds must move with them.
    let mut narrow_vision = ds_vision_config();
    narrow_vision.max_n_token = 32;
    let mut narrow = config(Some(narrow_vision));
    narrow.sliding_window = 8;
    let w2 = ImageWindow::for_chunk(&narrow, &tokens, 0)
        .unwrap()
        .unwrap();
    assert_ne!(w2.lo, w.lo, "a smaller window must shrink the left reach");
}

#[test]
fn pure_text_positions_reproduce_the_scalar_window_rule() {
    let tokens = expanded(&[(4, 6)]);
    let vocab = VOCAB as u32;
    let w = ImageWindow::for_chunk(&config(Some(ds_vision_config())), &tokens, 0)
        .unwrap()
        .unwrap();
    // The leading 40 text tokens sit before the first sentinel, so their
    // `(left, right)` are both 0 and the `_vis` kernel must behave exactly
    // like the scalar `(sliding_window, causal)` kernel there.
    for t in 0..40 {
        assert!(!block::is_sentinel(vocab, tokens[t]));
        assert_eq!(w.lo[t], (t as u32).saturating_sub(WINDOW - 1), "lo at {t}");
        assert_eq!(w.hi[t], t as u32, "hi at {t}");
    }
}

#[test]
fn a_suffix_chunk_and_zeroed_config_knobs_are_refused() {
    let tokens = expanded(&[(4, 6)]);
    let cfg = config(Some(ds_vision_config()));
    let err = ImageWindow::for_chunk(&cfg, &tokens, 1).unwrap_err();
    assert!(
        format!("{err}").contains("chunk-LOCAL"),
        "unexpected error: {err}"
    );

    let mut no_window = config(Some(ds_vision_config()));
    no_window.sliding_window = 0;
    assert!(ImageWindow::for_chunk(&no_window, &tokens, 0).is_err());

    let mut no_budget_vision = ds_vision_config();
    no_budget_vision.max_n_token = 0;
    assert!(ImageWindow::for_chunk(&config(Some(no_budget_vision)), &tokens, 0).is_err());

    // A text-only chunk at a non-zero start is fine — the refusal is scoped to
    // chunks that actually carry a span.
    let text: Vec<u32> = (1u32..=16).collect();
    assert!(ImageWindow::for_chunk(&cfg, &text, 512).unwrap().is_none());
}

#[test]
fn the_packed_pair_layout_matches_the_device_offsets() {
    let tokens = expanded(&[(3, 5)]);
    let w = ImageWindow::for_chunk(&config(Some(ds_vision_config())), &tokens, 0)
        .unwrap()
        .unwrap();
    let n = tokens.len();
    assert_eq!(w.hi_offset(), n * 4);

    let base = DevicePtr(0x1000);
    let (lo_ptr, hi_ptr) = w.ptrs(base);
    assert_eq!(lo_ptr.0, base.0);
    assert_eq!(hi_ptr.0, base.0 + (n * 4) as u64);

    // End to end through the real packer the prefill drivers use: pack both
    // halves at the offsets `ptrs` promises, then decode `packed()` back.
    let mut backing = vec![0u8; 16 + n * 8];
    // SAFETY: `backing` is live, zeroed and exclusively borrowed here; the
    // same stand-in for the zeroed pinned region `pinned_pack`'s tests use.
    let mut pack = unsafe { PinnedPacker::new(backing.as_mut_ptr(), backing.len()) };
    let lo_at = 16usize;
    pack.put_at("win_lo", lo_at, &w.lo).unwrap();
    pack.put_at("win_hi", lo_at + w.hi_offset(), &w.hi).unwrap();
    let packed = pack.packed();
    assert_eq!(packed.len(), lo_at + n * 8);
    let decode = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    assert_eq!(decode(&packed[lo_at..lo_at + n * 4]), w.lo);
    assert_eq!(decode(&packed[lo_at + n * 4..]), w.hi);
}
