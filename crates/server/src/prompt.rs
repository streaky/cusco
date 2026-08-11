use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
}

pub(crate) fn apply_chat_template(family: &str, messages: Vec<Message>) -> Result<String, Error> {
    match family {
        "gemma4" => gemma4(messages),
        "qwen35moe" => qwen35moe(messages),
        _ => Err(unsupported_template(family)),
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
            remove_generation_prompt(prompt, GENERATION_PROMPT, "Gemma")?;
            prompt.push_str(continuation);
            Ok(())
        }
        "qwen35moe" => {
            const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";
            remove_generation_prompt(prompt, GENERATION_PROMPT, "Qwen")?;
            prompt.push_str(continuation);
            Ok(())
        }
        _ => Err(unsupported_template(family)),
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
            require_generation_prompt(prompt, GENERATION_PROMPT, "Gemma")?;
            prompt.push_str(content);
            prompt.push_str("<end_of_turn>\n");
            prompt.push_str(GENERATION_PROMPT);
            Ok(())
        }
        "qwen35moe" => {
            const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";
            require_generation_prompt(prompt, GENERATION_PROMPT, "Qwen")?;
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
            prompt.push_str(GENERATION_PROMPT);
            Ok(())
        }
        _ => Err(unsupported_template(family)),
    }
}

pub(crate) fn insert_generation_instructions(
    family: &str,
    prompt: &mut String,
    instructions: &str,
) -> Result<(), Error> {
    let suffix = match family {
        "gemma4" => "<end_of_turn>\n<start_of_turn>model\n",
        "qwen35moe" => "<|im_end|>\n<|im_start|>assistant\n",
        _ => return Err(unsupported_template(family)),
    };
    let Some(position) = prompt.rfind(suffix) else {
        return Err(Error::State(format!(
            "rendered conversation for model family {family} lacks its generation suffix"
        )));
    };
    prompt.insert_str(position, instructions);
    Ok(())
}
pub(crate) fn terminal_markers(family: &str) -> &'static [&'static str] {
    match family {
        "gemma4" => &["<end_of_turn>", "</end_of_turn>", "</start_of_turn>"],
        "qwen35moe" => &["<|im_end|>", "<|endoftext|>"],
        _ => &[],
    }
}

fn unsupported_template(family: &str) -> Error {
    Error::BadRequest(format!(
        "unsupported_capability: model family {family} has no chat template"
    ))
}

fn require_generation_prompt(
    prompt: &str,
    generation_prompt: &str,
    family: &str,
) -> Result<(), Error> {
    if prompt.ends_with(generation_prompt) {
        Ok(())
    } else {
        Err(Error::State(format!(
            "rendered {family} conversation lacks its generation prompt"
        )))
    }
}

fn remove_generation_prompt(
    prompt: &mut String,
    generation_prompt: &str,
    family: &str,
) -> Result<(), Error> {
    require_generation_prompt(prompt, generation_prompt, family)?;
    prompt.truncate(prompt.len() - generation_prompt.len());
    Ok(())
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

fn qwen35moe(messages: Vec<Message>) -> Result<String, Error> {
    if messages.is_empty() {
        return Err(Error::BadRequest("messages must not be empty".into()));
    }
    if messages
        .iter()
        .any(|message| !matches!(message.role.as_str(), "system" | "user" | "assistant"))
    {
        return Err(Error::BadRequest(
            "Qwen chat supports only system, user, and assistant messages".into(),
        ));
    }

    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
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
    fn applies_qwen_template_and_supports_response_continuation() {
        let mut prompt = apply_chat_template(
            "qwen35moe",
            vec![
                Message {
                    role: "system".into(),
                    content: "Be concise.".into(),
                },
                Message {
                    role: "user".into(),
                    content: "Hello".into(),
                },
            ],
        )
        .unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );

        append_assistant_content("qwen35moe", &mut prompt, "Hi").unwrap();
        assert!(prompt.ends_with(
            "<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>assistant\n"
        ));
        insert_generation_instructions("qwen35moe", &mut prompt, "\nUse tools.").unwrap();
        assert!(prompt.contains("Hi\nUse tools.<|im_end|>"));
        append_rendered(
            "qwen35moe",
            &mut prompt,
            "<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n",
        )
        .unwrap();
        assert!(prompt.ends_with(
            "<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n"
        ));
        assert_eq!(
            terminal_markers("qwen35moe"),
            ["<|im_end|>", "<|endoftext|>"]
        );
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
