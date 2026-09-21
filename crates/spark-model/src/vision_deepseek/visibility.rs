// SPDX-License-Identifier: AGPL-3.0-only

//! Image-span attention bounds — the scalar port of `model.py::get_image_visible` and
//! `get_window_topk_idxs_visible`.
//!
//! The reference materialises a `[seqlen, min(seqlen, W + M)]` matrix of raw key indices
//! with `-1` for "masked". Atlas ships an inclusive `(lo, hi)` pair per query instead,
//! which is sound because every reference row is a contiguous interval — a property the
//! golden generator ASSERTS for every query of every stream and records as
//! `visible_contiguous`, so it is pinned by the reference rather than assumed here.

use super::block::{IMAGE_END, IMAGE_START, is_sentinel, sentinel_id};

/// Inclusive raw-key range per query, as indices within the prefilled chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowBounds {
    pub lo: Vec<u32>,
    pub hi: Vec<u32>,
}

/// `(left, right)` — distances to the enclosing span's START / END, 0 outside a span,
/// clamped to `max_image_tokens - 1` / `max_image_tokens`.
///
/// The reference builds these with a cumsum/cummax/cummin chain; the two scalar passes
/// below compute the same quantities.
pub fn image_visible(
    tokens: &[u32],
    vocab_size: u32,
    max_image_tokens: usize,
) -> (Vec<u32>, Vec<u32>) {
    let seqlen = tokens.len();
    let start_id = sentinel_id(vocab_size, IMAGE_START);
    let end_id = sentinel_id(vocab_size, IMAGE_END);
    let left_cap = max_image_tokens.saturating_sub(1) as u32;
    let right_cap = max_image_tokens as u32;

    let mut left = vec![0u32; seqlen];
    let mut right = vec![0u32; seqlen];
    let mut valid = vec![false; seqlen];

    // Forward pass:
    //   `valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end`  — inside an open span,
    //   or on the END that closes it;
    //   `starts = torch.where(is_start, idx, 0).cummax(1)[0]`      — index of the most
    //   recent START, or 0 before the first one;
    //   `left = (idx - starts) * valid`.
    let mut depth: i64 = 0;
    let mut last_start: u32 = 0;
    for (t, &tok) in tokens.iter().enumerate() {
        let is_start = tok == start_id;
        let is_end = tok == end_id;
        if is_start {
            depth += 1;
            last_start = t as u32;
        }
        if is_end {
            depth -= 1;
        }
        valid[t] = depth > 0 || is_end;
        if valid[t] {
            left[t] = (t as u32 - last_start).min(left_cap);
        }
    }

    // Backward pass:
    //   `ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)` — index of
    //   the next END at or after t, else `seqlen`;
    //   `right = (ends - idx) * valid`.
    let mut next_end = seqlen as u32;
    for t in (0..seqlen).rev() {
        if tokens[t] == end_id {
            next_end = t as u32;
        }
        if valid[t] {
            right[t] = (next_end - t as u32).min(right_cap);
        }
    }

    (left, right)
}

/// Per-query inclusive `(lo, hi)` raw-key bounds.
///
/// The reference's row is `starts` plus `0 .. width`, masked at `idx + right`, with
/// `width = min(seqlen, window_size + max_image_tokens)` and
/// `starts = (idx - (W - 1) - max(0, left - (W - 1))).clamp(min=0)`, so
///
/// * `lo = starts = t - max(W - 1, left)`, floored at 0, and
/// * `hi = min(t + right, lo + width - 1)`.
///
/// A final `hi = min(hi, seqlen - 1)` is applied on top. It is a NO-OP for every
/// well-formed stream — `right[t] = next_end[t] - t ≤ seqlen - 1 - t` whenever the span is
/// terminated — and only bites on a truncated stream whose span never closes, where
/// `next_end` degenerates to `seqlen`. That is what makes "`hi` indexes the chunk"
/// unconditional for the `_vis` attention kernel, which asserts it.
pub fn window_bounds(
    tokens: &[u32],
    vocab_size: u32,
    window: usize,
    max_image_tokens: usize,
) -> WindowBounds {
    let seqlen = tokens.len();
    // `config.sliding_window == 0` is a config bug the parser rejects; a floor of 1 keeps
    // this infallible function from underflowing rather than inventing a policy.
    let window = window.max(1);
    let (left, right) = image_visible(tokens, vocab_size, max_image_tokens);
    let width = seqlen.min(window + max_image_tokens) as u32;

    let mut lo = vec![0u32; seqlen];
    let mut hi = vec![0u32; seqlen];
    for t in 0..seqlen {
        let t32 = t as u32;
        let reach = ((window - 1) as u32).max(left[t]);
        lo[t] = t32.saturating_sub(reach);
        hi[t] = (t32 + right[t])
            .min(lo[t] + width.saturating_sub(1))
            .min(seqlen as u32 - 1);
    }
    WindowBounds { lo, hi }
}

/// True iff any token is a sentinel.
pub fn has_image_span(tokens: &[u32], vocab_size: u32) -> bool {
    tokens.iter().any(|&t| is_sentinel(vocab_size, t))
}

#[cfg(test)]
#[path = "visibility_tests.rs"]
mod tests;
