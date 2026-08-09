use serde::{Deserialize, Serialize};

pub const MAX_STOP_SEQUENCES: usize = 4;
pub const MAX_STOP_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopAlignment {
    TokenAligned,
    CrossToken,
    IntraToken,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierControl {
    Continue,
    Stop,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierResult {
    pub text: String,
    pub generated_tokens: usize,
    pub finish_reason: FinishReason,
    pub stop_alignment: Option<StopAlignment>,
}

pub struct GenerationFrontier {
    stops: Vec<Vec<u8>>,
    pending: Vec<u8>,
    pending_offset: usize,
    token_boundaries: Vec<usize>,
    sampled_tokens: usize,
    visible: String,
    stripping_leading_whitespace: bool,
    finish_reason: Option<FinishReason>,
    stop_alignment: Option<StopAlignment>,
}

impl GenerationFrontier {
    pub fn new(stops: &[String], raw_continuation: bool) -> Result<Self, &'static str> {
        if stops.len() > MAX_STOP_SEQUENCES {
            return Err("at most four stop sequences are supported");
        }
        if stops
            .iter()
            .any(|stop| stop.is_empty() || stop.len() > MAX_STOP_BYTES)
        {
            return Err("stop sequences must contain 1 to 256 UTF-8 bytes");
        }
        Ok(Self {
            stops: stops.iter().map(|stop| stop.as_bytes().to_vec()).collect(),
            pending: Vec::with_capacity(MAX_STOP_BYTES),
            pending_offset: 0,
            token_boundaries: vec![0],
            sampled_tokens: 0,
            visible: String::new(),
            stripping_leading_whitespace: !raw_continuation,
            finish_reason: None,
            stop_alignment: None,
        })
    }

    pub fn push(
        &mut self,
        piece: &[u8],
        terminal_or_control: bool,
        mut emit: impl FnMut(&str) -> Result<(), crate::Error>,
    ) -> Result<FrontierControl, crate::Error> {
        if self.finish_reason.is_some() {
            return Ok(FrontierControl::Stop);
        }
        self.sampled_tokens += 1;
        let token_start = self
            .token_boundaries
            .last()
            .copied()
            .unwrap_or(self.pending_offset);
        let token_end = token_start + piece.len();
        self.token_boundaries.push(token_end);
        if terminal_or_control {
            self.emit_prefix(self.pending.len(), true, &mut emit)?;
            self.finish_reason = Some(FinishReason::Stop);
            return Ok(FrontierControl::Stop);
        }
        self.pending.extend_from_slice(piece);
        if let Some((start, end)) = self.stop_match() {
            let absolute_start = self.pending_offset + start;
            let absolute_end = self.pending_offset + end;
            self.emit_prefix(start, true, &mut emit)?;
            self.stop_alignment = Some(self.classify_stop(absolute_start, absolute_end));
            self.pending.clear();
            self.pending_offset = absolute_end;
            self.finish_reason = Some(FinishReason::Stop);
            return Ok(FrontierControl::Stop);
        }
        let retained = self.longest_possible_stop_suffix();
        let safe = self.pending.len().saturating_sub(retained);
        self.emit_prefix(safe, false, &mut emit)?;
        self.token_boundaries
            .retain(|boundary| *boundary >= self.pending_offset);
        Ok(FrontierControl::Continue)
    }

    pub fn finish(
        mut self,
        mut emit: impl FnMut(&str) -> Result<(), crate::Error>,
    ) -> Result<FrontierResult, crate::Error> {
        if self.finish_reason.is_none() {
            self.emit_prefix(self.pending.len(), true, &mut emit)?;
            self.finish_reason = Some(FinishReason::Length);
        }
        Ok(FrontierResult {
            text: self.visible,
            generated_tokens: self.sampled_tokens,
            finish_reason: self.finish_reason.expect("finish reason assigned"),
            stop_alignment: self.stop_alignment,
        })
    }

    fn stop_match(&self) -> Option<(usize, usize)> {
        self.stops
            .iter()
            .enumerate()
            .filter_map(|(order, stop)| {
                self.pending
                    .windows(stop.len())
                    .position(|window| window == stop)
                    .map(|start| (start, order, start + stop.len()))
            })
            .min_by_key(|(start, order, _)| (*start, *order))
            .map(|(start, _, end)| (start, end))
    }

    fn longest_possible_stop_suffix(&self) -> usize {
        self.stops
            .iter()
            .map(|stop| {
                let maximum = stop.len().saturating_sub(1).min(self.pending.len());
                (1..=maximum)
                    .rev()
                    .find(|length| self.pending.ends_with(&stop[..*length]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }

    fn classify_stop(&self, start: usize, end: usize) -> StopAlignment {
        let starts_on_boundary = self.token_boundaries.contains(&start);
        let ends_on_boundary = self.token_boundaries.contains(&end);
        if starts_on_boundary && ends_on_boundary {
            StopAlignment::TokenAligned
        } else if self
            .token_boundaries
            .iter()
            .any(|boundary| start < *boundary && *boundary < end)
        {
            StopAlignment::CrossToken
        } else {
            StopAlignment::IntraToken
        }
    }

    fn emit_prefix(
        &mut self,
        requested: usize,
        final_flush: bool,
        emit: &mut impl FnMut(&str) -> Result<(), crate::Error>,
    ) -> Result<(), crate::Error> {
        let consumed = if final_flush {
            requested
        } else {
            utf8_complete_prefix(&self.pending[..requested])
        };
        if consumed == 0 {
            return Ok(());
        }
        let rendered = String::from_utf8_lossy(&self.pending[..consumed]);
        let delta = if self.stripping_leading_whitespace {
            let trimmed = rendered.trim_start_matches(char::is_whitespace);
            if !trimmed.is_empty() {
                self.stripping_leading_whitespace = false;
            }
            trimmed
        } else {
            rendered.as_ref()
        };
        if !delta.is_empty() {
            emit(delta)?;
            self.visible.push_str(delta);
        }
        self.pending.drain(..consumed);
        self.pending_offset += consumed;
        Ok(())
    }
}

fn utf8_complete_prefix(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        Err(_) => bytes.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[(&[u8], bool)], stops: &[&str], raw: bool) -> (FrontierResult, Vec<String>) {
        let stops = stops
            .iter()
            .map(|stop| (*stop).to_owned())
            .collect::<Vec<_>>();
        let mut frontier = GenerationFrontier::new(&stops, raw).unwrap();
        let mut deltas = Vec::new();
        for (piece, control) in pieces {
            if frontier
                .push(piece, *control, |delta| {
                    deltas.push(delta.to_owned());
                    Ok(())
                })
                .unwrap()
                == FrontierControl::Stop
            {
                break;
            }
        }
        let result = frontier
            .finish(|delta| {
                deltas.push(delta.to_owned());
                Ok(())
            })
            .unwrap();
        (result, deltas)
    }

    #[test]
    fn normalizes_whitespace_split_utf8_and_control_tokens() {
        let (result, deltas) = run(
            &[
                (b"  ", false),
                (&[0xf0, 0x9f], false),
                (&[0x98, 0x80], false),
                (b"ignored", true),
            ],
            &[],
            false,
        );
        assert_eq!(result.text, "😀");
        assert_eq!(deltas, ["😀"]);
        assert_eq!(result.generated_tokens, 4);
        assert_eq!(result.finish_reason, FinishReason::Stop);
        let (raw, _) = run(&[(b"  deliberate", false)], &[], true);
        assert_eq!(raw.text, "  deliberate");
    }

    #[test]
    fn classifies_token_cross_token_and_intra_token_stops() {
        type Fixture<'a> = (&'a [(&'a [u8], bool)], StopAlignment, &'a str);

        let fixtures: &[Fixture<'_>] = &[
            (
                &[(b"answer", false), (b"END", false)],
                StopAlignment::TokenAligned,
                "answer",
            ),
            (
                &[(b"answer E", false), (b"ND", false)],
                StopAlignment::CrossToken,
                "answer ",
            ),
            (
                &[(b"answerENDtail", false)],
                StopAlignment::IntraToken,
                "answer",
            ),
        ];
        for (pieces, expected_alignment, expected_text) in fixtures {
            let (result, _) = run(pieces, &["END"], false);
            assert_eq!(result.text, *expected_text);
            assert_eq!(result.stop_alignment, Some(*expected_alignment));
            assert_eq!(result.generated_tokens, pieces.len());
            assert_eq!(result.finish_reason, FinishReason::Stop);
        }
    }

    #[test]
    fn stop_wins_on_the_final_permitted_token_and_retains_that_token() {
        let (result, _) = run(
            &[(b"one", false), (b"STOP trailing", false)],
            &["STOP"],
            false,
        );
        assert_eq!(result.text, "one");
        assert_eq!(result.generated_tokens, 2);
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn validates_stop_bounds_and_flushes_possible_suffix_at_length() {
        assert!(GenerationFrontier::new(&vec!["x".into(); 5], false).is_err());
        assert!(GenerationFrontier::new(&[String::new()], false).is_err());
        assert!(GenerationFrontier::new(&["x".repeat(257)], false).is_err());
        let (result, _) = run(&[(b"possible ST", false)], &["STOP"], false);
        assert_eq!(result.text, "possible ST");
        assert_eq!(result.finish_reason, FinishReason::Length);
    }
}
