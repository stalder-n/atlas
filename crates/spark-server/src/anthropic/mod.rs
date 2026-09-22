// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic Messages API translation layer.
//!
//! Converts Anthropic `/v1/messages` requests to internal `InferenceRequest`,
//! reuses the existing scheduler pipeline, and converts the response back to
//! Anthropic format. Supports both streaming (SSE) and non-streaming.

mod handlers;
mod handlers_stream;
mod helpers;
mod to_ir;
mod translate;
mod translator;
// `pub(crate)` for ONE reason: `tokenizer/tests/deepseek_v4_vision.rs` (U2.4) lowers a real
// Anthropic wire body through `impl From<MessagesRequest> for ir::ChatRequest` to prove that
// an `image` block — top level and `tool_result`-nested — reaches the DeepSeek prompt path as
// a media ref with no adapter change. Nothing outside tests names this module.
pub(crate) mod types;

#[cfg(test)]
mod tests;

pub use handlers::{count_tokens, messages};
