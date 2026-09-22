// SPDX-License-Identifier: AGPL-3.0-only
//
// U2.4 `ds-server-prompt` — acceptance tests for the DeepSeek-V4 image prompt path:
// OpenAI `image_url` parts and Anthropic `image` blocks become ONE
// `<｜deepseek_image｜>` placeholder each, in client order, in a prompt byte-identical to
// DeepSeek's own `encoding/encoding_dsv4.py`; it then expands into
// `block::build_image_block` sentinel runs.
//
// ★ DECLARED FROM `api/chat/prepare.rs`, NOT `tokenizer/tests.rs`: that file is compiled
// by BOTH cargo targets and the `--lib` target has no `ir`/`api`/`anthropic`, so
// registering there breaks `cargo test -p spark-server --lib`. Precedent: the sibling
// `native_tool_prompt.rs`. Run with
// `cargo test -p spark-server --bin spark api::chat::prepare::deepseek_v4_vision`.
// Expected prompts are generated — see `gen_deepseek_v4_vision_goldens.py` (`--check`).

use std::path::Path;

use avarok_core::config::{VisionConfig, VisionFlavor};
use serde_json::{Value, json};
use spark_model::vision_deepseek::{block, geometry};

use crate::api::chat::prepare::test_lower_ds_messages;
use crate::ir::message::{ImageSource, ToolCall};
use crate::ir::{ChatRequest, ContentPart, ImageData, Message, Role, VideoSource};
use crate::tokenizer::ChatTokenizer;
use crate::tokenizer::deepseek_v4::encode_messages;

// ---- BEGIN GENERATED GOLDENS (gen_deepseek_v4_vision_goldens.py) ----
// Produced by DeepSeek's own `encoding/encoding_dsv4.py`, NOT hand-written.
//   repo:     deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
//   revision: 86f746b36186f0e567729a5c06a8c918caba82a9
//   source:   hf
//   sha256:   encoding_dsv4.py = b4bbb74bbb11a9c8ada04daa30cc7de7dba3abba08e9ade06d38b51a3d0d1701
//   generated:2026-09-21
//   invoked:  gen_deepseek_v4_vision_goldens.py
// Placeholder: <｜deepseek_image｜>
/// The reference's `IMAGE_PLACEHOLDER`, added-token id 129264.
pub(super) const PLACEHOLDER: &str = "<｜deepseek_image｜>";
/// G1 [chat, effort=None, images=1, tools=no] single image then text, chat mode (OpenAI `image_url` part)
const G1: &str =
    "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\ndescribe<｜Assistant｜></think>";
/// G2 [chat, effort=None, images=1, tools=no] interleaved text/image/text, in the ATLAS-NORMALISED form the IR can represent (image first, the two text parts concatenated) — accepted deviation
const G2: &str =
    "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\nbeforeafter<｜Assistant｜></think>";
/// G3 [chat, effort=None, images=2, tools=no] two images in one turn, then text
const G3: &str = "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>";
/// G4 [chat, effort=None, images=1, tools=yes] image inside a tool result (the reference's own tool-role image case)
const G4: &str = "<｜begin▁of▁sentence｜>You are helpful.\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"screenshot\", \"description\": \"take a screenshot\", \"parameters\": {\"type\": \"object\", \"properties\": {\"region\": {\"type\": \"string\"}}}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>grab a shot<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"screenshot\">\n<｜DSML｜parameter name=\"region\" string=\"true\">full</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result><｜deepseek_image｜>\n\ntool image</tool_result><｜Assistant｜></think>";
/// G5 [chat, effort=None, images=1, tools=no] Anthropic `image` block — `to_ir` lowers it to the same images-then-text IR
const G5: &str = "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\nanthropic look<｜Assistant｜></think>";
/// G6 [thinking, effort=low, images=1, tools=no] image on an ASSISTANT turn, thinking mode with `low` effort
const G6: &str = "<｜begin▁of▁sentence｜><｜User｜>first look<｜Assistant｜></think><｜deepseek_image｜>\n\nI see a cat<｜end▁of▁sentence｜><｜User｜>and now?<｜Assistant｜><think>";
// ---- END GENERATED GOLDENS ----

/// Solid 64x64 and 128x64 PNGs. Pixels never reach a prompt assertion; the SIZES are what
/// the geometry test reads, and 128x64 keeps the two-image case from passing by symmetry.
const PNG_64X64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAUUlEQVR42u3PMQ0AMAgAMJgRZGIMgVPBQdI6aFZPXPbiOAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAYF9H3PtAcqGlQhkAAAAAElFTkSuQmCC";
const PNG_128X64: &str = "iVBORw0KGgoAAAANSUhEUgAAAIAAAABACAIAAABdtOgoAAAAdElEQVR42u3RQQEAMAjEsGNmwL8Y9EwGn1RCU9sT3fUsAABAAAAIAAABACAAAAQAgAAAEAAAAgBAAAAIAAABACAAAAQAgAAAEAAAAgBAAAAIAAABACAAAAQAgAAAEAAAAgBAAAAIAAABACAAAAQAgAAAUPIBn74BpWma37QAAAAASUVORK5CYII=";

const VOCAB_SIZE: u32 = 129_280;
const PLACEHOLDER_ID: u32 = 129_264;

fn data_uri(b64: &str) -> String {
    format!("data:image/png;base64,{b64}")
}

/// The checkpoint's own `vision_*` values, in the shape the DS flat-key parser builds.
fn ds_vision_config() -> VisionConfig {
    VisionConfig {
        depth: 32,
        hidden_size: 1024,
        num_heads: 16,
        patch_size: 14,
        temporal_patch_size: 1,
        spatial_merge_size: 3, // vision_downsample_ratio
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
    }
}

fn msg(role: Role, content: Vec<ContentPart>) -> Message {
    Message {
        role,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn txt(s: &str) -> ContentPart {
    ContentPart::Text(s.to_string())
}

fn img(b64: &str) -> ContentPart {
    ContentPart::Image(ImageSource {
        data: ImageData::Base64(data_uri(b64)),
    })
}

/// Images first, then the text — what `media_kinds()`/`text()` and `parts_from` produce.
fn user_images_then_text(images: &[&str], text: &str) -> Message {
    let mut parts: Vec<ContentPart> = images.iter().map(|b| img(b)).collect();
    parts.push(txt(text));
    msg(Role::User, parts)
}

fn screenshot_tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "screenshot", "description": "take a screenshot",
        "parameters": {"type": "object", "properties": {"region": {"type": "string"}}}}}])
}

/// The G4 conversation: a tool call whose RESULT carries an image.
fn tool_result_image_messages() -> Vec<Message> {
    let mut assistant = msg(Role::Assistant, vec![txt("")]);
    let (id, name) = ("call-1".to_string(), "screenshot".to_string());
    let arguments = json!({"region": "full"});
    assistant.tool_calls = vec![ToolCall {
        id,
        name,
        arguments,
    }];
    let mut tool = msg(Role::Tool, vec![img(PNG_64X64), txt("tool image")]);
    tool.tool_call_id = Some("call-1".into());
    let head = [
        msg(Role::System, vec![txt("You are helpful.")]),
        msg(Role::User, vec![txt("grab a shot")]),
    ];
    head.into_iter().chain([assistant, tool]).collect()
}

fn anthropic_ir(body: Value) -> ChatRequest {
    serde_json::from_value::<crate::anthropic::types::MessagesRequest>(body)
        .expect("MessagesRequest")
        .into()
}

/// C8's rendering half: `build_msg_entries` (media collection + DeepSeek image
/// preprocessing) → `build_json_messages_ds` → `encode_messages(.., ds_vision)`.
fn ds_prompt(
    messages: &[Message],
    tools_active: bool,
    tools: Option<&Value>,
    thinking: bool,
    effort: Option<&str>,
) -> String {
    let cfg = ds_vision_config();
    let lowered = test_lower_ds_messages(Some(&cfg), messages, tools_active)
        .expect("DeepSeek lowering must accept these messages");
    let tools = tools.map(|t| t.as_array().expect("tools array").as_slice());
    encode_messages(&lowered.json_messages, tools, thinking, effort, true)
        .expect("DeepSeek-V4 vision encoding")
}

fn assert_exact(actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let (a, e) = (actual.bytes(), expected.bytes());
    let at = a
        .zip(e)
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let win = |s: &str| {
        s[s.floor_char_boundary(at.saturating_sub(80))
            ..s.floor_char_boundary(s.len().min(at + 120))]
            .to_string()
    };
    panic!(
        "DeepSeek-V4 vision prompt mismatch at byte {at} (len {} vs {})\nactual:   {:?}\nexpected: {:?}",
        actual.len(),
        expected.len(),
        win(actual),
        win(expected),
    );
}

/// A minimal `tokenizer.json`: a WordLevel model whose `Whitespace` pre-tokenizer shreds
/// `<｜deepseek_image｜>` into three unknown pieces, plus (optionally) the checkpoint's
/// ADDED TOKEN for it at 129264 with `"special": false`. The whole C8 design rests on HF
/// `tokenizers` pulling a NON-special added token out as ONE id under
/// `add_special_tokens = false` — that asymmetry is what makes the negative half of the
/// round-trip test a real assertion rather than a tautology. The placeholder is ALSO in
/// the model vocab because `tokenizers` 0.23.2 ignores an `added_tokens` entry's `id` (it
/// warns, then assigns `model.get_vocab_size()`) — and a vocab entry alone still shreds.
fn write_tokenizer(dir: &Path, with_placeholder: bool) {
    let added = if with_placeholder {
        json!([{"id": PLACEHOLDER_ID, "content": "<｜deepseek_image｜>", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": false}])
    } else {
        json!([])
    };
    let spec = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab":
            {"<unk>": 0, "hello": 1, "world": 2, "<｜deepseek_image｜>": PLACEHOLDER_ID}}
    });
    std::fs::write(dir.join("tokenizer.json"), spec.to_string()).expect("write tokenizer.json");
}

fn tokenizer_at(dir: &Path) -> ChatTokenizer {
    ChatTokenizer::from_model_dir(dir, 1, true, "deepseek_v4", None, true)
        .expect("synthetic DeepSeek tokenizer loads")
}

/// A tokenizer with the DS image path armed at `config.vocab_size`.
fn armed(dir: &Path) -> ChatTokenizer {
    let mut tk = tokenizer_at(dir);
    let id = tk.arm_ds_vision(VOCAB_SIZE).expect("placeholder resolves");
    assert_eq!(id, PLACEHOLDER_ID);
    tk
}

/// THE UNIT'S CENTRAL CLAIM: for every shape a client can send, the prompt Atlas renders
/// for a DeepSeek vision checkpoint is BYTE-IDENTICAL to `encoding_dsv4.py`'s at the
/// pinned revision. G2 pins the ACCEPTED DEVIATION — the IR cannot represent interleaving
/// (`MsgEntry` is `content: String` + `media: Vec<MediaKind>`), so `[text, image, text]`
/// renders as `[image, "beforeafter"]`. Weakening any of these to "contains the
/// placeholder" would hide a real regression in the `"\n\n"` join.
#[test]
fn rendered_prompts_match_the_reference() {
    let one = [user_images_then_text(&[PNG_64X64], "describe")];
    assert_exact(&ds_prompt(&one, false, None, false, None), G1);

    let parts = vec![txt("before"), img(PNG_64X64), txt("after")];
    let interleaved = [msg(Role::User, parts)];
    assert_exact(&ds_prompt(&interleaved, false, None, false, None), G2);

    let two = [user_images_then_text(&[PNG_64X64, PNG_128X64], "compare")];
    assert_exact(&ds_prompt(&two, false, None, false, None), G3);

    let tools = screenshot_tools();
    let tool_case = tool_result_image_messages();
    assert_exact(&ds_prompt(&tool_case, true, Some(&tools), false, None), G4);

    let seen = vec![img(PNG_64X64), txt("I see a cat")];
    let turns = [
        msg(Role::User, vec![txt("first look")]),
        msg(Role::Assistant, seen),
        msg(Role::User, vec![txt("and now?")]),
    ];
    assert_exact(&ds_prompt(&turns, false, None, true, Some("low")), G6);
}

/// VERIFICATION THAT `anthropic/*` NEEDS NO CHANGE: a top-level `image` block and a
/// `tool_result`-nested one already reach `msg_entry` as media refs via
/// `to_ir::parts_from`, so the Anthropic surface lands on the same prompt bytes.
#[test]
fn anthropic_image_blocks_reach_the_same_prompt() {
    let image = json!({"type": "image", "source":
        {"type": "base64", "media_type": "image/png", "data": PNG_64X64}});

    let simple = anthropic_ir(json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user",
            "content": [image, {"type": "text", "text": "anthropic look"}]}]
    }));
    assert_exact(&ds_prompt(&simple.messages, false, None, false, None), G5);

    let nested = anthropic_ir(json!({
        "model": "m", "max_tokens": 16, "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "grab a shot"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1",
                "name": "screenshot", "input": {"region": "full"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1",
                "content": [image, {"type": "text", "text": "tool image"}]}]}
        ]
    }));
    let tools = screenshot_tools();
    let got = ds_prompt(&nested.messages, true, Some(&tools), false, None);
    assert_exact(&got, G4);
}

/// The DS branch must run the DeepSeek geometry, never Qwen's smart-resize:
/// `VisionItem.grid_*` are `plan_image`'s ViT patch counts, the patch buffer is
/// `n_vit_h · n_vit_w · 3 · p · p` long, and `image_pad_counts` stays ZERO (a Qwen
/// spatial-merge count would drag the prompt through `expand_vision_pads`).
#[test]
fn ds_image_items_use_the_deepseek_geometry_and_no_qwen_pads() {
    let cfg = ds_vision_config();
    let g = geometry::DsVisionGeometry::from_vision_config(&cfg).expect("DS geometry");
    let two = [user_images_then_text(&[PNG_64X64, PNG_128X64], "compare")];
    let lowered = test_lower_ds_messages(Some(&cfg), &two, false).expect("two DS images");

    assert_eq!(lowered.image_pixels.len(), 2, "one VisionItem per image");
    assert_eq!(lowered.image_pad_counts, vec![0, 0], "no Qwen pad counts");
    let sizes = [(64usize, 64usize), (128, 64)];
    for (item, (w, h)) in lowered.image_pixels.iter().zip(sizes) {
        let plan = geometry::plan_image(w, h, &g).expect("plan");
        assert_eq!(item.groups.len(), 1, "a still image has one temporal group");
        assert_eq!((item.grid_h, item.grid_w), (plan.n_vit_h, plan.n_vit_w));
        assert_eq!(
            item.groups[0].len(),
            plan.n_vit_h * plan.n_vit_w * 3 * g.patch_size * g.patch_size
        );
    }
}

/// The DS tower has `temporal_patch_size = 1` and no video path, and the reference's
/// `_is_image_block` cannot see a `{"type":"video"}` marker — so a video part would
/// silently render ONE FEWER placeholder than there are media items. 400 instead.
#[test]
fn video_on_a_ds_tower_is_rejected() {
    let clip = ContentPart::Video(VideoSource {
        data: ImageData::Base64(data_uri(PNG_64X64)),
    });
    let video = [msg(Role::User, vec![clip, txt("what happens")])];
    let why = test_lower_ds_messages(Some(&ds_vision_config()), &video, false)
        .expect_err("a video part on a DeepSeek vision checkpoint must be refused");
    // Not an incidental "Video decode error" from the Qwen decoder: an explicit refusal.
    assert!(why.contains("does not accept video input"), "{why}");
}

/// PCND at boot: `arm_ds_vision` resolves `<｜deepseek_image｜>` through the server's own
/// tokenizer and demands exactly one id — it does not trust the `special` flag, and fails
/// the serve rather than let a multi-token placeholder reach `expand_placeholders`.
#[test]
fn placeholder_round_trips_to_one_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_tokenizer(dir.path(), true);
    let tk = armed(dir.path());
    assert!(tk.ds_vision_armed());
    assert_eq!(
        tk.encode("<｜deepseek_image｜>").expect("encode"),
        vec![PLACEHOLDER_ID],
        "a NON-special added token must still split out as one id"
    );

    let bare = tempfile::tempdir().expect("tempdir");
    write_tokenizer(bare.path(), false);
    let mut unarmed = tokenizer_at(bare.path());
    assert!(
        unarmed.arm_ds_vision(VOCAB_SIZE).is_err(),
        "a tokenizer without the placeholder must fail the serve at boot"
    );
    assert!(!unarmed.ds_vision_armed());
}

/// `expand_ds_image_blocks` is a thin, SSOT-preserving wrapper: equal to
/// `block::expand_placeholders` with the armed placeholder + vocab_size. That pins the
/// SEQUENTIAL start positions — an image's compress pad depends on how many tokens were
/// already emitted, so block two's length is not independent of block one's.
#[test]
fn expansion_matches_block_expand_placeholders() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_tokenizer(dir.path(), true);
    let tk = armed(dir.path());
    let ph = PLACEHOLDER_ID;
    let tokens = vec![1, 2, ph, 3, ph, 4];
    let grids = [(2usize, 3usize), (4, 4)];

    let got = tk
        .expand_ds_image_blocks(tokens.clone(), &grids)
        .expect("expansion");
    let (want, _) = block::expand_placeholders(&tokens, ph, VOCAB_SIZE, &grids).expect("reference");
    assert_eq!(got, want, "the wrapper must not re-derive the layout");

    let len0 = block::block_len(2, 3, 2);
    let len1 = block::block_len(4, 4, 2 + len0 + 1);
    assert_eq!(got.len(), tokens.len() - 2 + len0 + len1);
    assert!(
        got.iter()
            .filter(|t| **t >= VOCAB_SIZE)
            .all(|t| block::sentinel_type(VOCAB_SIZE, *t).is_some()),
        "every id at or above vocab_size must be one of the five sentinels"
    );
    assert!(!got.contains(&ph), "no bare placeholder survives expansion");
}

/// The count check is the designed failure mode for every placeholder↔image desync (a
/// dropped message, a compaction, a placeholder smuggled through a tool schema).
#[test]
fn placeholder_count_mismatch_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_tokenizer(dir.path(), true);
    let tk = armed(dir.path());
    let ph = PLACEHOLDER_ID;

    let bad = [
        (vec![1, ph, 2], &[(2usize, 3usize), (4, 4)][..]), // one placeholder, two images
        (vec![1, ph, ph, 2], &[(2, 3)][..]),               // two placeholders, one image
        (vec![1, ph, 2], &[][..]),                         // a smuggled bare placeholder
    ];
    for (tokens, grids) in bad {
        let got = tk.expand_ds_image_blocks(tokens.clone(), grids);
        assert!(got.is_err(), "{tokens:?} must not expand");
    }
    // The zero-image scan runs on every DS request and must leave text alone.
    let inert = tk.expand_ds_image_blocks(vec![1, 2, 3], &[]);
    assert_eq!(inert.expect("no images"), vec![1, 2, 3]);
}

/// Port of `_validate_no_image_sp_tokens` plus the text-block guard: a client may not
/// hand-write the placeholder, in `content`, in `reasoning_content`, or in a text block.
/// Sentinels are >= `vocab_size`, outside `lm_head`'s range: a smuggled one is a crash.
#[test]
fn user_supplied_placeholder_is_rejected() {
    let in_content = json!([{"role": "user", "content": format!("look {PLACEHOLDER} here")}]);
    let in_reasoning = json!([
        {"role": "assistant", "content": "ok", "reasoning_content": PLACEHOLDER},
        {"role": "user", "content": "hi"}
    ]);
    let in_block = json!([{"role": "user", "content": [{"type": "text", "text": PLACEHOLDER}]}]);

    for case in [&in_content, &in_reasoning, &in_block] {
        assert!(
            encode_messages(case.as_array().unwrap(), None, false, None, true).is_err(),
            "armed: a user-supplied placeholder must be a 400, not a prompt: {case}"
        );
    }
    // Inert on a TEXT checkpoint: the flag arms the validation, so a 0731 serve keeps
    // accepting a string that happens to contain the literal.
    assert!(
        encode_messages(in_content.as_array().unwrap(), None, false, None, false).is_ok(),
        "the validation must not fire when the checkpoint has no vision tower"
    );
}

/// A DS checkpoint WITHOUT vision renders an image marker rather than failing with
/// "unexpected DeepSeek-V4 content item", and must never emit the placeholder.
#[test]
fn image_marker_on_a_text_checkpoint_renders_a_marker() {
    let msgs = json!([{"role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]);
    let out = encode_messages(msgs.as_array().unwrap(), None, false, None, false)
        .expect("a bare image marker must render, not error");
    assert!(out.contains("[Image]"), "got {out:?}");
    assert!(!out.contains(PLACEHOLDER), "got {out:?}");
}

/// REGRESSION GUARD for the DeepSeek-V4 support-matrix entry: the four official fixtures
/// contain no images, so the new `ds_vision` flag must be completely inert — BOTH values
/// reproduce the committed reference output byte-for-byte.
#[test]
fn text_fixtures_are_byte_identical_with_the_flag_either_way() {
    const IN: [&str; 4] = [
        include_str!("../../../test_data/deepseek_v4_encoding/test_input_1.json"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_input_2.json"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_input_3.json"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_input_4.json"),
    ];
    const OUT: [&str; 4] = [
        include_str!("../../../test_data/deepseek_v4_encoding/test_output_1.txt"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_output_2.txt"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_output_3.txt"),
        include_str!("../../../test_data/deepseek_v4_encoding/test_output_4.txt"),
    ];
    // Fixture 4 is the non-thinking case, matching `tests/deepseek_v4.rs`.
    const THINKING: [bool; 4] = [true, true, true, false];
    for ((input, expected), thinking) in IN.into_iter().zip(OUT).zip(THINKING) {
        let fixture: Value = serde_json::from_str(input).expect("official fixture JSON");
        let (messages, tools) = match fixture.as_object() {
            Some(object) => (
                object["messages"].as_array().expect("messages").clone(),
                object
                    .get("tools")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            ),
            None => (fixture.as_array().expect("message array").clone(), vec![]),
        };
        let tools = (!tools.is_empty()).then_some(tools.as_slice());
        for ds_vision in [false, true] {
            let got = encode_messages(&messages, tools, thinking, Some("low"), ds_vision)
                .expect("official encoding");
            assert_exact(&got, expected);
        }
    }
}

/// The real 6.4 MB `tokenizer.json` is not committable, so the shipped placeholder id is
/// pinned behind `DSV4_TOKENIZER_DIR` (the U1.3 `$DSV4_VISION_CKPT` precedent).
#[test]
#[ignore = "needs the real checkpoint; set DSV4_TOKENIZER_DIR"]
fn real_checkpoint_placeholder_is_129264() {
    let dir = std::env::var("DSV4_TOKENIZER_DIR").expect("DSV4_TOKENIZER_DIR");
    let dir = Path::new(&dir);
    let mut tk = ChatTokenizer::from_model_dir(dir, 1, true, "deepseek_v4", None, true)
        .expect("real tokenizer loads");
    let id = tk.arm_ds_vision(VOCAB_SIZE).expect("placeholder resolves");
    assert_eq!(id, PLACEHOLDER_ID);
}
