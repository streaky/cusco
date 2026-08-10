use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
}

pub(crate) fn apply_chat_template(family: &str, messages: Vec<Message>) -> Result<String, Error> {
    match family {
        "gemma4" => gemma4(messages),
        _ => Err(Error::BadRequest(format!(
            "unsupported_capability: model family {family} has no chat template"
        ))),
    }
}
pub(crate) fn append_rendered(
    family: &str,
    prompt: &mut String,
    continuation: &str,
) -> Result<(), Error> {
    if prompt.is_empty() {
        prompt.push_str(continuation);
        return Ok(());
    }
    match family {
        "gemma4" => {
            const GENERATION_PROMPT: &str = "<start_of_turn>model\n";
            if !prompt.ends_with(GENERATION_PROMPT) {
                return Err(Error::State(
                    "rendered Gemma conversation lacks its generation prompt".into(),
                ));
            }
            prompt.truncate(prompt.len() - GENERATION_PROMPT.len());
            prompt.push_str(continuation);
            Ok(())
        }
        _ => Err(Error::BadRequest(format!(
            "unsupported_capability: model family {family} has no chat template"
        ))),
    }
}
pub(crate) fn append_assistant_content(
    family: &str,
    prompt: &mut String,
    content: &str,
) -> Result<(), Error> {
    match family {
        "gemma4" => {
            const GENERATION_PROMPT: &str = "<start_of_turn>model\n";
            if !prompt.ends_with(GENERATION_PROMPT) {
                return Err(Error::State(
                    "rendered Gemma conversation lacks its generation prompt".into(),
                ));
            }
            prompt.push_str(content);
            prompt.push_str("<end_of_turn>\n");
            prompt.push_str(GENERATION_PROMPT);
            Ok(())
        }
        _ => Err(Error::BadRequest(format!(
            "unsupported_capability: model family {family} has no chat template"
        ))),
    }
}

pub(crate) fn insert_generation_instructions(
    family: &str,
    prompt: &mut String,
    instructions: &str,
) -> Result<(), Error> {
    match family {
        "gemma4" => {
            const GENERATION_SUFFIX: &str = "<end_of_turn>\n<start_of_turn>model\n";
            let Some(position) = prompt.rfind(GENERATION_SUFFIX) else {
                return Err(Error::State(
                    "rendered Gemma conversation lacks its generation suffix".into(),
                ));
            };
            prompt.insert_str(position, instructions);
            Ok(())
        }
        _ => Err(Error::BadRequest(format!(
            "unsupported_capability: model family {family} has no chat template"
        ))),
    }
}
pub(crate) fn terminal_markers(family: &str) -> &'static [&'static str] {
    match family {
        "gemma4" => &["<end_of_turn>", "</end_of_turn>", "</start_of_turn>"],
        _ => &[],
    }
}

fn gemma4(mut messages: Vec<Message>) -> Result<String, Error> {
    if messages.is_empty() {
        return Err(Error::BadRequest("messages must not be empty".into()));
    }

    let system = if messages
        .first()
        .is_some_and(|message| message.role == "system")
    {
        Some(messages.remove(0).content)
    } else {
        None
    };
    if messages.iter().any(|message| message.role == "system") {
        return Err(Error::BadRequest(
            "system messages are accepted only in the first position".into(),
        ));
    }
    if messages.is_empty() || messages[0].role != "user" {
        return Err(Error::BadRequest(
            "Gemma chat must begin with a user message".into(),
        ));
    }
    if let Some(system) = system {
        messages[0].content = format!("{system}\n\n{}", messages[0].content);
    }

    let mut prompt = String::new();
    for (index, message) in messages.into_iter().enumerate() {
        let expected = if index % 2 == 0 { "user" } else { "assistant" };
        if message.role != expected {
            return Err(Error::BadRequest(format!(
                "Gemma chat messages must alternate user and assistant roles; expected {expected}"
            )));
        }
        let template_role = if message.role == "assistant" {
            "model"
        } else {
            "user"
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(template_role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_gemma_template_once_and_maps_roles() {
        let prompt = apply_chat_template(
            "gemma4",
            vec![
                Message {
                    role: "system".into(),
                    content: "Be concise.".into(),
                },
                Message {
                    role: "user".into(),
                    content: "Hello".into(),
                },
                Message {
                    role: "assistant".into(),
                    content: "Hi".into(),
                },
                Message {
                    role: "user".into(),
                    content: "Again".into(),
                },
            ],
        )
        .unwrap();
        assert_eq!(
            prompt,
            "<start_of_turn>user\nBe concise.\n\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>\n<start_of_turn>user\nAgain<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(prompt.matches("<start_of_turn>model\n").count(), 2);
    }

    #[test]
    fn rejects_non_alternating_and_tool_messages() {
        let error = apply_chat_template(
            "gemma4",
            vec![Message {
                role: "tool".into(),
                content: "result".into(),
            }],
        )
        .unwrap_err();
        assert!(matches!(error, Error::BadRequest(_)));
    }
}
