// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::collections::HashMap;

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn last_user_index(messages: &[Value]) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| matches!(role(m), Some("user" | "developer")))
}

pub(super) fn attach_top_level_tools(
    messages: &mut Vec<Value>,
    tools: Option<&[Value]>,
) -> Result<()> {
    let Some(tools) = tools.filter(|v| !v.is_empty()) else {
        return Ok(());
    };
    if messages.iter().any(|m| {
        m.get("tools")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
    }) {
        return Ok(());
    }

    let tools = Value::Array(tools.to_vec());
    let recipient = messages
        .iter()
        .position(|m| role(m) == Some("system"))
        .or_else(|| messages.iter().position(|m| role(m) == Some("developer")));
    if let Some(index) = recipient {
        let message = &mut messages[index];
        message
            .as_object_mut()
            .context("DeepSeek-V4 message must be an object")?
            .insert("tools".into(), tools);
    } else {
        messages.insert(0, json!({"role": "system", "content": "", "tools": tools}));
    }
    Ok(())
}

pub(super) fn merge_tool_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        let message_role = role(message).unwrap_or_default();
        match message_role {
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                });
                if let Some(blocks) = merged.last_mut().and_then(user_content_blocks_mut) {
                    blocks.push(block);
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            "user" => {
                let content = message
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                let block = json!({"type": "text", "text": content});
                let can_merge = merged.last().is_some_and(|m| {
                    role(m) == Some("user")
                        && m.get("content_blocks").and_then(Value::as_array).is_some()
                        && m.get("task").is_none()
                });
                if can_merge {
                    merged
                        .last_mut()
                        .and_then(user_content_blocks_mut)
                        .expect("checked content_blocks")
                        .push(block);
                } else {
                    let mut new_message = message.clone();
                    new_message
                        .as_object_mut()
                        .context("DeepSeek-V4 user message must be an object")?
                        .insert("content_blocks".into(), Value::Array(vec![block]));
                    merged.push(new_message);
                }
            }
            _ => merged.push(message.clone()),
        }
    }
    Ok(merged)
}

fn user_content_blocks_mut(message: &mut Value) -> Option<&mut Vec<Value>> {
    (role(message) == Some("user"))
        .then(|| message.get_mut("content_blocks")?.as_array_mut())
        .flatten()
}

pub(super) fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut call_order: HashMap<String, usize> = HashMap::new();
    for message in messages {
        match role(message) {
            Some("assistant") => {
                let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
                    continue;
                };
                if tool_calls.is_empty() {
                    continue;
                }
                call_order.clear();
                for (index, call) in tool_calls.iter().enumerate() {
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .or_else(|| {
                            call.get("function")
                                .and_then(|v| v.get("id"))
                                .and_then(Value::as_str)
                        })
                        .unwrap_or_default();
                    if !id.is_empty() {
                        call_order.insert(id.to_string(), index);
                    }
                }
            }
            Some("user") if !call_order.is_empty() => {
                let Some(blocks) = message
                    .get_mut("content_blocks")
                    .and_then(Value::as_array_mut)
                else {
                    continue;
                };
                let mut results: Vec<Value> = blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                    .cloned()
                    .collect();
                if results.len() < 2 {
                    continue;
                }
                results.sort_by_key(|b| {
                    b.get("tool_use_id")
                        .and_then(Value::as_str)
                        .and_then(|id| call_order.get(id).copied())
                        .unwrap_or(0)
                });
                let mut results = results.into_iter();
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        *block = results.next().expect("same number of result blocks");
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let Some(last_user) = last_user_index(messages) else {
        return messages.to_vec();
    };
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        if index >= last_user
            || matches!(
                role(message),
                // `developer` is load-bearing here: msg_entry lowers it
                // deliberately for this encoder (preserve_developer_role), so
                // omitting it from the keep-list silently DELETED developer
                // instructions from the prompt, with no error.
                Some(
                    "user"
                        | "system"
                        | "developer"
                        | "tool"
                        | "latest_reminder"
                        | "direct_search_results"
                )
            )
        {
            out.push(message.clone());
            continue;
        }
        if let Some("assistant") = role(message) {
            let mut message = message.clone();
            if let Some(object) = message.as_object_mut() {
                object.remove("reasoning_content");
            }
            out.push(message);
        }
    }
    out
}

// ─────────────────────────── vision: image → placeholder ───────────────────────────
//
// A port of the reference's `process_image_messages` / `_process_image_blocks` /
// `_validate_no_image_sp_tokens` (`encoding/encoding_dsv4.py`, revision
// 86f746b36186f0e567729a5c06a8c918caba82a9).
//
// ★ WHY HERE AND NOT IN `render::content_text`. The reference substitutes the
// placeholder BEFORE the text blocks are joined, and joins them with `"\n\n"`;
// `content_text` joins array parts with `""`. Substituting there would render
// `<｜User｜><｜deepseek_image｜>describe` where the checkpoint was trained on
// `<｜User｜><｜deepseek_image｜>\n\ndescribe`. Running the port at the head of
// `encode_messages`, ahead of `merge_tool_messages`, reproduces the reference bytes
// for free — the `"\n\n"`-join of the text blocks and the single joined `content`
// string that `merge_tool_messages` re-wraps are the same bytes.
//
// ★ `_extract_image` IS DELIBERATELY NOT PORTED. Atlas never carries image bytes
// inside the JSON message array: the binding is the ordered `Vec<VisionItem>` that
// `msg_entry::collect_message_media` builds by walking `m.content` in client order,
// while `template::build_json_messages_ds` emits one `{"type":"image"}` marker per
// media item in that same order. The i-th placeholder in the rendered prompt is
// therefore the i-th `VisionItem` by construction, and the count equality is enforced
// downstream by `block::expand_placeholders`.
//
// ★ DIVERGENCE, INERT TODAY: Atlas's `merge_tool_messages` ignores a pre-existing
// `content_blocks` on a user message and always re-wraps `content` into a single text
// block, where the reference honours it. Because the pass below collapses
// `content_blocks` back into a `"\n\n"`-joined `content` string, both produce identical
// bytes for every shape Atlas can generate — but a caller that supplied
// `content_blocks` directly would see the difference.

/// The reference's `IMAGE_PLACEHOLDER`. Resolves to added-token id 129264 in the
/// shipped tokenizer; the id itself is never hard-coded (see
/// `ChatTokenizer::arm_ds_vision`).
pub(crate) const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// Replace every image content block with `{"type":"text","text": IMAGE_PLACEHOLDER}`,
/// in place and in order, and rebuild each touched message's `content` string the way
/// the reference does.
///
/// Only called when the serve is armed for a DeepSeek vision checkpoint, so a text
/// checkpoint's bytes — and its tolerance for a string that merely contains the
/// literal — are untouched.
pub(super) fn process_image_messages(messages: &mut [Value]) -> Result<()> {
    for message in messages.iter_mut() {
        validate_no_image_sp_tokens(message)?;
        let object = message
            .as_object_mut()
            .context("DeepSeek-V4 message must be an object")?;

        // `content` as a list is the OpenAI spelling of `content_blocks`.
        if object.get("content").is_some_and(Value::is_array)
            && !object.contains_key("content_blocks")
        {
            let content = object.remove("content").expect("checked present above");
            object.insert("content_blocks".into(), content);
        }

        let Some(blocks) = object
            .get_mut("content_blocks")
            .and_then(Value::as_array_mut)
            .filter(|blocks| !blocks.is_empty())
        else {
            continue;
        };
        process_image_blocks(blocks)?;
        let joined = join_text_blocks(blocks);
        // The reference rebuilds `content` only when it is not already a string —
        // i.e. exactly when it was moved into `content_blocks` above.
        if !object.get("content").is_some_and(Value::is_string) {
            object.insert("content".into(), Value::String(joined));
        }
    }
    Ok(())
}

/// `"\n\n".join(block["text"] for top-level text blocks)` — the reference's list
/// comprehension. Nested `tool_result` text never participates.
fn join_text_blocks(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .map(|b| b.get("text").and_then(Value::as_str).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// One ordered traversal, recursing into `tool_result.content` exactly as the
/// reference does (the tool-result image is the motivating case). Recursion depth is
/// bounded by serde_json's own nesting limit on the request body.
fn process_image_blocks(blocks: &mut [Value]) -> Result<()> {
    for block in blocks.iter_mut() {
        // `map(str::to_string)` rather than a borrow: the image arm replaces `*block`.
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string);
        match kind.as_deref() {
            Some("image" | "image_url") => {
                *block = json!({"type": "text", "text": IMAGE_PLACEHOLDER});
            }
            Some("tool_result") => {
                if let Some(nested) = block.get_mut("content").and_then(Value::as_array_mut) {
                    process_image_blocks(nested)?;
                }
            }
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if text.contains(IMAGE_PLACEHOLDER) {
                    let shown: String = text.chars().take(100).collect();
                    bail!(
                        "text block contains the image placeholder '{IMAGE_PLACEHOLDER}': \
                         '{shown}'. Images must be sent as separate image content blocks."
                    );
                }
            }
            // Anything else — including `{"type":"video"}`, which `msg_entry` has
            // already refused for this tower — passes through untouched.
            _ => {}
        }
    }
    Ok(())
}

/// Port of `_validate_no_image_sp_tokens`: a client may not hand-write the
/// placeholder. Its expansion produces sentinel ids at or above `vocab_size`, which
/// `lm_head` cannot represent and the embedding splice would read out of range.
fn validate_no_image_sp_tokens(message: &Value) -> Result<()> {
    if message
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|c| c.contains(IMAGE_PLACEHOLDER))
    {
        bail!(
            "message content contains the image special token '{IMAGE_PLACEHOLDER}'. \
             Images should be provided as image content blocks."
        );
    }
    if message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|c| c.contains(IMAGE_PLACEHOLDER))
    {
        bail!("reasoning_content contains the image special token '{IMAGE_PLACEHOLDER}'");
    }
    Ok(())
}
