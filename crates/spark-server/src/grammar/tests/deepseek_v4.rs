// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::tool_parser::{DeepseekV4DsmlParser, ToolCallParser};
use xgrammar::GrammarMatcher;

fn compile(schema: serde_json::Value, auto: bool) -> xgrammar::CompiledGrammar {
    let mut tools = test_tool_defs();
    tools[0].function.name = "run".into();
    tools[0].function.parameters = Some(schema);
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    DeepseekV4DsmlParser
        .compile_tool_grammar(&mut engine, &tools, auto)
        .expect("DSML must register a grammar")
        .unwrap()
}

fn accepts(grammar: &xgrammar::CompiledGrammar, text: &str) -> bool {
    let mut matcher = GrammarMatcher::new(grammar, None, true, -1).unwrap();
    matcher.accept_string(text, false) && matcher.is_terminated()
}

fn envelope(params: &str) -> String {
    format!(
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"run\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
}

#[test]
fn dsml_strings_preserve_empty_unicode_code_and_whitespace() {
    let grammar = compile(
        serde_json::json!({"type":"object", "properties":{"q":{"type":"string"}}, "required":["q"], "additionalProperties":false}),
        false,
    );
    for value in ["", "你好🌍", "\n  =x => \"<div>\";\n", "</other>"] {
        let text = envelope(&format!(
            "<｜DSML｜parameter name=\"q\" string=\"true\">{value}</｜DSML｜parameter>"
        ));
        assert!(accepts(&grammar, &text), "{text:?}");
        assert!(!accepts(
            &grammar,
            &text.replace("string=\"true\"", "string=\"false\"")
        ));
    }
    assert!(!accepts(&grammar, &envelope("")));
    assert!(!accepts(
        &grammar,
        &envelope("<｜DSML｜parameter name=\"unknown\" string=\"true\">x</｜DSML｜parameter>")
    ));
}

#[test]
fn dsml_typed_values_and_multiple_invocations() {
    for (schema, value) in [
        (serde_json::json!({"type":"integer"}), "42"),
        (serde_json::json!({"type":"boolean"}), "false"),
        (serde_json::json!({"type":"null"}), "null"),
        (
            serde_json::json!({"type":"array","items":{"type":"string"}}),
            "[\"你好\",\"x\"]",
        ),
        (
            serde_json::json!({"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}),
            "{\"a\":true}",
        ),
    ] {
        let grammar = compile(
            serde_json::json!({"type":"object", "properties":{"q":schema}, "required":["q"], "additionalProperties":false}),
            false,
        );
        let text = envelope(&format!(
            "<｜DSML｜parameter name=\"q\" string=\"false\">{value}</｜DSML｜parameter>"
        ));
        assert!(accepts(&grammar, &text), "{text:?}");
        assert!(!accepts(
            &grammar,
            &text.replace("string=\"false\"", "string=\"true\"")
        ));
        assert!(!accepts(
            &grammar,
            &text.replace("name=\"run\"", "name=\"unknown\"")
        ));
        let invoke = text
            .strip_prefix("<｜DSML｜tool_calls>\n")
            .unwrap()
            .strip_suffix("\n</｜DSML｜tool_calls>")
            .unwrap();
        assert!(accepts(
            &grammar,
            &format!("<｜DSML｜tool_calls>\n{invoke}\n{invoke}\n</｜DSML｜tool_calls>")
        ));
    }
}

#[test]
fn dsml_optional_union_enum_and_malformed_envelopes() {
    let grammar = compile(
        serde_json::json!({"type":"object", "properties":{
        "q":{"type":["string","integer"]},
        "z":{"enum":["a\n\"b", false]}
    }, "required":["q"], "additionalProperties":false}),
        false,
    );
    let parameter = "<｜DSML｜parameter name=\"q\" string=\"false\">7</｜DSML｜parameter>";
    let text = envelope(parameter);
    assert!(accepts(&grammar, &text));
    assert!(accepts(
        &grammar,
        &envelope(&format!(
            "{parameter}\n<｜DSML｜parameter name=\"z\" string=\"true\">a\n\"b</｜DSML｜parameter>"
        ))
    ));
    assert!(!accepts(
        &grammar,
        &envelope(&format!("{parameter}\n{parameter}"))
    ));
    for malformed in [
        text.replace("</｜DSML｜invoke>", "</invoke>"),
        text.replace("</｜DSML｜parameter>", "</parameter>"),
        text.replace("string=\"false\"", "string=\"maybe\""),
        "<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>".into(),
    ] {
        assert!(!accepts(&grammar, &malformed), "{malformed:?}");
    }
    let auto = compile(serde_json::json!({"type":"object", "properties":{}}), true);
    assert!(accepts(&auto, "Plain answer without a tool."));
    assert!(accepts(
        &auto,
        &format!("<think>considering</think>{}", envelope(""))
    ));
}

#[test]
fn dsml_masks_tokens_after_split_delimiter() {
    let mut vocab = test_vocab();
    // No token contains the entire opening delimiter.
    for token in [
        "<｜DS",
        "ML｜tool_",
        "calls>",
        "</｜DSML｜tool_calls>",
        "\n<｜DSML｜invoke name=\"run\">\n",
    ] {
        vocab.push(token.into());
    }
    let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
    let mut tools = test_tool_defs();
    tools[0].function.name = "run".into();
    let compiled = DeepseekV4DsmlParser
        .compile_tool_grammar(&mut engine, &tools, true)
        .unwrap()
        .unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, false, -1).unwrap();
    let mut mask = vec![0i32; vocab.len().div_ceil(32)];
    let allowed = |mask: &[i32], token: usize| mask[token / 32] & (1 << (token % 32)) != 0;
    for token in 131..=133 {
        matcher.fill_next_token_bitmask(&mut mask, 0, false);
        assert!(allowed(&mask, token));
        assert!(matcher.accept_token(token as i32));
    }
    matcher.fill_next_token_bitmask(&mut mask, 0, false);
    let accepts_close = matcher.accept_token(134);
    assert!(
        !accepts_close,
        "matcher accepted closing envelope after split trigger"
    );
    assert!(!allowed(&mask, 134), "an empty envelope must be masked");
    assert!(
        allowed(&mask, 135),
        "the registered invocation remains legal"
    );
    assert!(
        !allowed(&mask, 130),
        "EOS cannot end an incomplete envelope"
    );
}

#[test]
fn dsml_production_grammar_state_masks_split_envelope() {
    let mut vocab = test_vocab();
    for token in [
        "<｜DS",
        "ML｜tool_",
        "calls>",
        "</｜DSML｜tool_calls>",
        "\n<｜DSML｜invoke name=\"run\">\n",
    ] {
        vocab.push(token.into());
    }
    let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
    let mut tools = test_tool_defs();
    tools[0].function.name = "run".into();
    tools[0].function.parameters = Some(
        serde_json::json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}),
    );
    let compiled = DeepseekV4DsmlParser
        .compile_tool_grammar(&mut engine, &tools, true)
        .unwrap()
        .unwrap();
    let mut state = GrammarState::new(&compiled, vocab.len()).unwrap();
    for token in 131..=133 {
        state.fill_bitmask();
        assert!(state.is_token_allowed(token));
        assert!(state.accept_token(token));
    }
    state.fill_bitmask();
    assert!(!state.is_token_allowed(134));
    assert!(!state.is_token_allowed(130));
    assert!(state.is_token_allowed(135));
}

#[test]
fn dsml_attribute_names_are_round_trippable() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let mut tools = test_tool_defs();
    for name in ["bad\"name", "bad\nname"] {
        tools[0].function.name = name.into();
        assert!(
            engine
                .compile_deepseek_v4_tool_grammar(&tools, false)
                .is_err()
        );
    }
    tools[0].function.name = "查找".into();
    tools[0].function.parameters =
        Some(serde_json::json!({"type":"object","properties":{"bad\"name":{"type":"string"}}}));
    assert!(
        engine
            .compile_deepseek_v4_tool_grammar(&tools, false)
            .is_err()
    );
    tools[0].function.parameters = Some(
        serde_json::json!({"type":"object","properties":{"查询":{"type":"string"}},"required":["查询"]}),
    );
    let compiled = engine
        .compile_deepseek_v4_tool_grammar(&tools, false)
        .unwrap();
    assert!(accepts(
        &compiled,
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"查找\">\n<｜DSML｜parameter name=\"查询\" string=\"true\">你好</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    ));
}
