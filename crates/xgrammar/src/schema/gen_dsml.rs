// SPDX-License-Identifier: AGPL-3.0-only

//! DSML's string attribute is a type discriminator, not a free choice.
//! Keep the existing object/schema machinery, but include that discriminator
//! in each top-level parameter value rule. Nested values remain JSON.

use super::converter::{JsonSchemaConverter, XML_STRING};
use super::error::{SchemaError, SchemaResult};
use super::spec::{SchemaSpec, SchemaSpecPtr, SpecKind};

impl JsonSchemaConverter<'_> {
    pub(super) fn create_dsml_value_rule(
        &mut self,
        spec: &SchemaSpecPtr,
        hint: &str,
    ) -> SchemaResult<String> {
        let name = self.script.allocate_rule_name(hint);
        let body = self.dsml_value_body(spec, &name)?;
        self.script.add_rule_with_allocated_name(&name, &body);
        Ok(name)
    }

    fn dsml_value_body(&mut self, spec: &SchemaSpecPtr, hint: &str) -> SchemaResult<String> {
        match &spec.kind {
            SpecKind::AnyOf(options) | SpecKind::TypeArray(options) => {
                let alternatives = options
                    .iter()
                    .map(|option| self.create_dsml_value_rule(option, hint))
                    .collect::<SchemaResult<Vec<_>>>()?;
                Ok(alternatives.join(" | "))
            }
            SpecKind::Enum(values) => {
                let alternatives = values
                    .iter()
                    .map(|value| {
                        self.dsml_value_body(
                            &SchemaSpec::make(SpecKind::Const(value.clone()), "", ""),
                            hint,
                        )
                    })
                    .collect::<SchemaResult<Vec<_>>>()?;
                Ok(alternatives.join(" | "))
            }
            SpecKind::AllOf(options) if options.len() == 1 => {
                self.dsml_value_body(&options[0], hint)
            }
            SpecKind::AllOf(_) => Err(SchemaError::invalid(
                "DSML does not support multi-schema allOf",
            )),
            SpecKind::Ref(uri) => {
                if self.dsml_refs.contains(uri) || self.dsml_refs.len() >= 64 {
                    return Err(SchemaError::invalid(
                        "DSML cyclic or excessively deep parameter reference",
                    ));
                }
                let resolved = self.parser.resolve_ref(uri, hint)?;
                self.dsml_refs.push(uri.clone());
                let body = self.dsml_value_body(&resolved, hint);
                self.dsml_refs.pop();
                body
            }
            SpecKind::Any => Ok(format!(
                r#""true\">" {XML_STRING} | "false\">" (basic_number | basic_boolean | basic_null | basic_array | basic_object)"#
            )),
            SpecKind::String(_) => {
                let raw = self.generate_from_spec(spec, hint)?;
                Ok(format!(r#""true\">" ({raw})"#))
            }
            SpecKind::Const(value) if value.starts_with('"') => {
                let raw: String = serde_json::from_str(value)
                    .map_err(|error| SchemaError::invalid(error.to_string()))?;
                let literal = serde_json::to_string(&raw)
                    .map_err(|error| SchemaError::invalid(error.to_string()))?;
                Ok(format!(r#""true\">" {literal}"#))
            }
            _ => {
                // Use the JSON cache and JSON string rules for every non-string
                // value, including nested arrays/objects and constant values.
                self.nested_object_level += 1;
                let value = self.create_rule(spec, hint);
                self.nested_object_level -= 1;
                Ok(format!(r#""false\">" {}"#, value?))
            }
        }
    }
}
