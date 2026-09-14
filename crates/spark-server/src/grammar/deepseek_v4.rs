// SPDX-License-Identifier: AGPL-3.0-only

//! Native DSML envelope and schema-constrained invocation bodies.

use super::{GrammarEngine, GrammarError};
use crate::tool_parser::{DSML_CLOSE, DSML_OPEN, ToolDefinition};
use xgrammar::CompiledGrammar;

impl GrammarEngine {
    pub fn compile_deepseek_v4_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }
        let mut invocations = Vec::with_capacity(tools.len());
        for tool in tools {
            if tool
                .function
                .name
                .chars()
                .any(|c| c == '"' || c.is_control())
            {
                return Err(GrammarError::InvalidSchema(
                    "tool name cannot be represented in a DSML attribute".into(),
                ));
            }
            let schema = tool.function.parameters.clone().unwrap_or_else(|| {
                serde_json::json!({"type":"object", "properties":{}, "additionalProperties":false})
            });
            // Keep string contents and schema constraints intact. In particular,
            // required strings may be empty; the shared nonempty-string policy
            // used by other formats is not part of DSML.
            let body = xgrammar::deepseek_xml_tool_calling_to_ebnf(&schema.to_string())
                .map_err(|error| GrammarError::InvalidSchema(error.to_string()))?;
            invocations.push(serde_json::json!({
                "type":"tag",
                "begin":format!("<｜DSML｜invoke name=\"{}\">\n", tool.function.name),
                "content":{"type":"grammar", "grammar":body},
                "end":"\n</｜DSML｜invoke>",
            }));
        }
        let envelope = serde_json::json!({
            "type":"tag", "begin":format!("{DSML_OPEN}\n"),
            "content":{
                "type":"tags_with_separator", "tags":invocations,
                "separator":"\n", "at_least_one":true, "stop_after_first":false,
            },
            "end":format!("\n{DSML_CLOSE}"),
        });
        self.compile_structural_tag_raw(
            &[DSML_OPEN.into()],
            &[envelope],
            !use_triggers,
            !use_triggers,
        )
    }
}
