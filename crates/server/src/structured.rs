use serde_json::{Map, Value};
use thiserror::Error;

const JSON_GRAMMAR: &str = r#"root ::= value space
value ::= object | array | string | number | boolean | null
object ::= "{" space (string space ":" space value (space "," space string space ":" space value)*)? space "}"
array ::= "[" space (value (space "," space value)*)? space "]"
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
space ::= | " " | "\n" [ \t]{0,20}
"#;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid JSON Schema: {0}")]
    Invalid(String),
    #[error("unsupported JSON Schema feature at {path}: {feature}")]
    Unsupported { path: String, feature: &'static str },
}

pub fn json_grammar() -> String {
    JSON_GRAMMAR.into()
}

pub fn schema_grammar(schema: &Value) -> Result<String, Error> {
    jsonschema::JSONSchema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .compile(schema)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let mut compiler = Compiler {
        rules: vec!["space ::= | \" \" | \"\\n\" [ \\t]{0,20}".into()],
        next_rule: 0,
    };
    let value = compiler.compile(schema, "$")?;
    compiler.rules.insert(0, format!("root ::= {value} space"));
    Ok(compiler.rules.join("\n") + "\n")
}

struct Compiler {
    rules: Vec<String>,
    next_rule: usize,
}

impl Compiler {
    fn compile(&mut self, schema: &Value, path: &str) -> Result<String, Error> {
        let object = schema
            .as_object()
            .ok_or_else(|| Error::Invalid(format!("{path} must be a schema object")))?;
        for keyword in object.keys() {
            if !matches!(
                keyword.as_str(),
                "type"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "items"
                    | "enum"
                    | "const"
                    | "description"
                    | "title"
                    | "$schema"
            ) {
                return Err(Error::Unsupported {
                    path: path.into(),
                    feature: "schema keyword",
                });
            }
        }
        if let Some(value) = object.get("const") {
            return Ok(literal(value));
        }
        if let Some(values) = object.get("enum") {
            let values = values
                .as_array()
                .ok_or_else(|| Error::Invalid(format!("{path}.enum must be an array")))?;
            if values.is_empty() {
                return Err(Error::Invalid(format!("{path}.enum must not be empty")));
            }
            return Ok(values.iter().map(literal).collect::<Vec<_>>().join(" | "));
        }
        match object.get("type").and_then(Value::as_str) {
            Some("object") => self.object(object, path),
            Some("array") => self.array(object, path),
            Some("string") => Ok("json-string".into()),
            Some("integer") => Ok("integer".into()),
            Some("number") => Ok("number".into()),
            Some("boolean") => Ok("boolean".into()),
            Some("null") => Ok("null".into()),
            Some(other) => Err(Error::Invalid(format!(
                "{path}.type has unknown value {other}"
            ))),
            None => Err(Error::Unsupported {
                path: path.into(),
                feature: "schemas without an explicit scalar type",
            }),
        }
    }

    fn object(&mut self, object: &Map<String, Value>, path: &str) -> Result<String, Error> {
        if object.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(Error::Unsupported {
                path: path.into(),
                feature: "objects must set additionalProperties to false",
            });
        }
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Invalid(format!("{path}.properties must be an object")))?;
        let required = object
            .get("required")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Unsupported {
                path: path.into(),
                feature: "every object property must be required",
            })?;
        if required.len() != properties.len()
            || properties.keys().any(|name| {
                !required
                    .iter()
                    .any(|required| required.as_str() == Some(name))
            })
        {
            return Err(Error::Unsupported {
                path: path.into(),
                feature: "every object property must be required",
            });
        }
        let mut members = Vec::with_capacity(properties.len());
        for (name, property) in properties {
            let value = self.compile(property, &format!("{path}.properties.{name}"))?;
            members.push(format!(
                "{} space \":\" space ({value})",
                literal(&Value::String(name.clone()))
            ));
        }
        let body = if members.is_empty() {
            "\"{\" space \"}\"".into()
        } else {
            format!(
                "\"{{\" space {} space \"}}\"",
                members.join(" space \",\" space ")
            )
        };
        Ok(self.rule("object", body))
    }

    fn array(&mut self, object: &Map<String, Value>, path: &str) -> Result<String, Error> {
        let items = object
            .get("items")
            .ok_or_else(|| Error::Invalid(format!("{path}.items is required")))?;
        let item = self.compile(items, &format!("{path}.items"))?;
        Ok(self.rule(
            "array",
            format!("\"[\" space (({item}) (space \",\" space ({item}))*)? space \"]\""),
        ))
    }

    fn rule(&mut self, prefix: &str, body: String) -> String {
        let name = format!("{prefix}-{}", self.next_rule);
        self.next_rule += 1;
        self.rules.push(format!("{name} ::= {body}"));
        name
    }
}

fn literal(value: &Value) -> String {
    let json = serde_json::to_string(value).expect("JSON value serializes");
    format!("\"{}\"", json.replace('\\', "\\\\").replace('"', "\\\""))
}

// Shared scalar rules are appended once to schema grammars by callers.
pub fn finish_schema_grammar(mut grammar: String) -> String {
    grammar.push_str(
        "json-string ::= \"\\\"\" ([^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\"\ninteger ::= \"-\"? ([0-9] | [1-9] [0-9]*)\nnumber ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\nboolean ::= \"true\" | \"false\"\nnull ::= \"null\"\n",
    );
    grammar
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compiles_strict_nested_schema_deterministically() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}}
            },
            "required": ["answer", "confidence", "tags"],
            "additionalProperties": false
        });
        let first = finish_schema_grammar(schema_grammar(&schema).unwrap());
        let second = finish_schema_grammar(schema_grammar(&schema).unwrap());
        assert_eq!(first, second);
        assert!(first.contains("\\\"answer\\\""));
        assert!(first.contains("array-"));
    }

    #[test]
    fn rejects_optional_properties_explicitly() {
        let error = schema_grammar(&json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
            "additionalProperties": false
        }))
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("every object property must be required")
        );
    }

    #[test]
    fn rejects_invalid_schema() {
        assert!(matches!(
            schema_grammar(&json!({"type": "not-a-type"})),
            Err(Error::Invalid(_))
        ));
    }
}
