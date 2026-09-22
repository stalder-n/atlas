// SPDX-License-Identifier: AGPL-3.0-only

//! Host side of the DeepSeek-V4 image-span attention bounds (C6).
//!
//! A prompt that carries image sentinel blocks is not causal inside those
//! blocks: every token of a span sees every other token of the same span,
//! while text around it stays causal. That is a PER-QUERY key interval, not a
//! scalar window, so it cannot be expressed through the scalar kernel's
//! `sliding_window` argument — hence `prefill_attn_compressed_vis`, which
//! reads an inclusive `(win_lo[q], win_hi[q])` pair instead.
//!
//! This module turns a chunk's token ids into that pair, on the host, with no
//! GPU and no I/O. The arithmetic itself is SSOT in
//! [`crate::vision_deepseek::visibility::window_bounds`]; what lives here is
//! the config gate, the invariants the kernel relies on, and the device-offset
//! layout of the packed pair.

use anyhow::{Result, ensure};
use avarok_core::config::{ModelConfig, VisionFlavor};
use spark_runtime::gpu::DevicePtr;

use crate::vision_deepseek::visibility;

/// Per-query INCLUSIVE raw-key bounds for one V4 prefill chunk, host side.
///
/// `lo`/`hi` are indices WITHIN the chunk, both `[chunk_len]`, and satisfy
/// `lo[t] <= hi[t] < chunk_len` — the contract
/// `prefill_attn_compressed_vis` assumes (its in-kernel clamp is a
/// memory-safety backstop against a host bug, not a policy).
#[derive(Debug)]
pub(crate) struct ImageWindow {
    pub(crate) lo: Vec<u32>,
    pub(crate) hi: Vec<u32>,
}

impl ImageWindow {
    /// `None` — the overwhelmingly common case — when this chunk carries no
    /// image sentinel, or the model is not a DeepSeek-V4 vision checkpoint.
    /// Callers then leave [`crate::layer::ForwardContext::image_window`]
    /// `None` and every existing kernel path stays byte-identical.
    ///
    /// `chunk_start` is the absolute position of `tokens[0]` in the sequence.
    /// A non-zero start with a span present is an ERROR rather than a
    /// silently-wrong bound: the emitted indices are chunk-local, so a suffix
    /// chunk would mis-locate every span it contains. Chunked prefill and
    /// prefix-cached prefixes are out of scope for image prompts; the server
    /// rejects them, and this is the loud statement of that invariant.
    pub(crate) fn for_chunk(
        config: &ModelConfig,
        tokens: &[u32],
        chunk_start: usize,
    ) -> Result<Option<Self>> {
        let Some(vision) = config
            .vision
            .as_ref()
            .filter(|v| v.flavor == VisionFlavor::DeepseekV4)
        else {
            return Ok(None);
        };
        let vocab = u32::try_from(config.vocab_size)?;
        if !visibility::has_image_span(tokens, vocab) {
            return Ok(None);
        }
        ensure!(
            chunk_start == 0,
            "DeepSeek-V4 image-span prefill: the chunk starting at token {chunk_start} contains \
             image sentinels, but the attention bounds are chunk-LOCAL indices — a suffix chunk \
             would mis-locate every span. Image prompts must be prefilled in a single pass \
             (no chunked prefill, no prefix-cache reuse across a sentinel block)."
        );
        // PCND: the window and the image-token budget are config, not literals,
        // and a checkpoint that declares neither must fail rather than default.
        ensure!(
            config.sliding_window > 0,
            "DeepSeek-V4 image-span prefill: config.sliding_window is 0; the image-span key \
             bounds are derived from it and there is no defensible default."
        );
        ensure!(
            vision.max_n_token > 0,
            "DeepSeek-V4 image-span prefill: config.vision.max_n_token is 0; the image-span key \
             bounds are derived from it and there is no defensible default."
        );
        let bounds = visibility::window_bounds(
            tokens,
            vocab,
            config.sliding_window as usize,
            vision.max_n_token,
        );
        // The kernel's host contract, pinned once per chunk. `window_bounds`
        // already guarantees both (`hi` carries a `min(seqlen - 1)` clamp, and
        // `hi >= lo` because every `min` operand is `>= lo`), so this is a pin,
        // not a fix-up: a regression there must surface here, not as a wrong
        // attention result on the Spark.
        for t in 0..tokens.len() {
            ensure!(
                bounds.lo[t] <= bounds.hi[t] && (bounds.hi[t] as usize) < tokens.len(),
                "DeepSeek-V4 image-span prefill: bounds for query {t} are \
                 ({}, {}) with chunk_len {} — the _vis kernel requires lo <= hi < chunk_len.",
                bounds.lo[t],
                bounds.hi[t],
                tokens.len()
            );
        }
        Ok(Some(Self {
            lo: bounds.lo,
            hi: bounds.hi,
        }))
    }

    /// Byte offset of `win_hi` relative to `win_lo` in the packed pair.
    pub(crate) fn hi_offset(&self) -> usize {
        self.lo.len() * 4
    }

    /// `(win_lo, win_hi)` for a packed pair based at `base`.
    pub(crate) fn ptrs(&self, base: DevicePtr) -> (DevicePtr, DevicePtr) {
        (base, base.offset(self.hi_offset()))
    }
}

#[cfg(test)]
#[path = "v4_visibility_tests.rs"]
mod tests;
