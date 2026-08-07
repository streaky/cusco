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
