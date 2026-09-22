// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4 vision half of `impl ChatTokenizer`.
//!
//! Three jobs, all of them C8's:
//!   1. resolve `<｜deepseek_image｜>` through the checkpoint's OWN tokenizer at boot
//!      and demand that it round-trips to exactly one id (PCND — a multi-token
//!      placeholder would desync every image block, so the serve fails instead);
//!   2. remember the `config.vocab_size` the sentinel ids are derived from, so the
//!      request path never re-derives it;
//!   3. expand a rendered token stream through
//!      [`spark_model::vision_deepseek::block::expand_placeholders`], which stays the
//!      single source of truth for the image-token layout.
//!
//! A separate file from `chat_impl.rs` purely because that file sits at the 500-LoC cap.

use anyhow::{Result, bail};

use spark_model::vision_deepseek::block;

use super::{ChatEncoding, ChatTokenizer, deepseek_v4::IMAGE_PLACEHOLDER};

/// What arming the DeepSeek image path resolves. Both fields come from the serve's
/// own state — the id from this tokenizer, the vocab size from `ModelConfig` after
/// every cap has been applied — so nothing downstream carries a literal.
#[derive(Clone, Copy)]
pub(crate) struct DsVisionTokens {
    /// The `<｜deepseek_image｜>` token id (129264 in the shipped tokenizer).
    pub(crate) placeholder: u32,
    /// `config.vocab_size`: sentinel `ty` is `vocab_size + ty`.
    pub(crate) vocab_size: u32,
}

impl ChatTokenizer {
    /// Arm the DeepSeek image prompt path for a vision checkpoint, returning the
    /// resolved placeholder id.
    ///
    /// MUST be called after every `config.vocab_size` cap has run
    /// (`cap_vocab_size_to_tokenizer`, `resolve_tokenizer_runtime`): the sentinel base
    /// is that number, and reading it early would silently shift all five ids.
    pub fn arm_ds_vision(&mut self, vocab_size: u32) -> Result<u32> {
        if self.chat_encoding != ChatEncoding::DeepseekV4 {
            bail!(
                "the DeepSeek image prompt path requires the checkpoint-native \
                 DeepSeek-V4 encoding, got {:?}",
                self.chat_encoding
            );
        }
        if vocab_size == 0 {
            bail!("config.vocab_size must be > 0 to derive the DeepSeek image sentinel ids");
        }
        let ids = self.encode(IMAGE_PLACEHOLDER)?;
        let [placeholder] = ids[..] else {
            bail!(
                "this checkpoint's tokenizer encodes the image placeholder \
                 '{IMAGE_PLACEHOLDER}' as {} ids ({ids:?}), not one — it is not a \
                 DeepSeek-V4 vision tokenizer",
                ids.len()
            );
        };
        self.ds_vision = Some(DsVisionTokens {
            placeholder,
            vocab_size,
        });
        Ok(placeholder)
    }

    /// True once [`Self::arm_ds_vision`] has succeeded. The request path's flavor
    /// switch; set from the same boot site as `config.vision.flavor`.
    pub fn ds_vision_armed(&self) -> bool {
        self.ds_vision.is_some()
    }

    /// Replace every placeholder id with its image's sentinel block.
    ///
    /// A thin wrapper — the layout itself is `block::build_image_block`'s, reached
    /// through `expand_placeholders` so the start positions stay SEQUENTIAL (an
    /// image's compress pad depends on how many tokens were already emitted).
    ///
    /// Called on EVERY armed request, including those with no image: it is also the
    /// guard that a bare placeholder id — smuggled through a tool schema, which the
    /// reference's `_validate_no_image_sp_tokens` does not scan — can never reach the
    /// model. The blocks are discarded here; the model rebuilds them from the token
    /// stream (SSOT).
    pub fn expand_ds_image_blocks(
        &self,
        tokens: Vec<u32>,
        grids: &[(usize, usize)],
    ) -> Result<Vec<u32>> {
        let Some(armed) = self.ds_vision else {
            bail!("the DeepSeek image prompt path is not armed for this checkpoint");
        };
        let (expanded, _blocks) =
            block::expand_placeholders(&tokens, armed.placeholder, armed.vocab_size, grids)?;
        Ok(expanded)
    }
}
