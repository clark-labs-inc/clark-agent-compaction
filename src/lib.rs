//! Provider-agnostic context compaction primitives for agent transcripts.
//!
//! The crate owns the deterministic parts of compaction:
//! threshold checks, transcript rendering budgets, recent user-message
//! retention, and summary handoff finalization. Callers own provider I/O,
//! persistence, cancellation, telemetry, and conversion back into their typed
//! message model.

mod config;
mod plain;
mod truncate;

pub use config::{
    CharHeuristic, CompactionConfig, TokenEstimator, DEFAULT_AUTO_COMPACT_TOKEN_LIMIT,
    DEFAULT_COMPACTION_PROMPT, DEFAULT_COMPACT_REQUEST_TOKEN_LIMIT,
    DEFAULT_RECENT_USER_TOKEN_BUDGET, DEFAULT_SUMMARY_PREFIX,
};
pub use plain::{PlainMessage, PlainToolCall};
pub use truncate::{truncate_to_token_budget, truncate_to_token_budget_with_estimator};

/// Runtime-owned transcript adapter.
///
/// Implement this trait for the caller's typed message enum. Rendering should
/// be deterministic, concise, and model-readable.
pub trait TranscriptMessage {
    /// Render the message into the summarization transcript.
    fn render_for_compaction(&self, out: &mut String);

    /// Write the user-visible text for real user messages.
    ///
    /// Return `false` for non-user messages. Images or binary attachments should
    /// be omitted or represented by safe caller-owned text.
    fn user_text_for_compaction(&self, out: &mut String) -> bool;

    /// Identify compaction-summary messages that should not be retained as real
    /// recent user messages.
    fn is_compaction_summary(&self, summary_prefix: &str) -> bool {
        let mut text = String::new();
        self.user_text_for_compaction(&mut text) && text.starts_with(summary_prefix)
    }
}

/// Caller-owned model request and deterministic replacement plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedCompaction {
    pub request: CompactionRequest,
    pub plan: CompactionPlan,
}

/// Prompt to send to the caller's summarization model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionRequest {
    pub prompt: String,
    pub omitted_messages: usize,
    pub estimated_transcript_tokens: usize,
    pub estimated_request_tokens: usize,
}

/// Deterministic data needed to install a model-produced summary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionPlan {
    pub recent_user_messages: Vec<String>,
    pub summary_prefix: String,
}

/// Final model-visible compaction output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactedTranscript {
    pub summary_message: String,
    pub recent_user_messages: Vec<String>,
}

/// Estimate the rendered transcript token count.
pub fn estimate_transcript_tokens<M, E>(messages: &[M], estimator: &E) -> usize
where
    M: TranscriptMessage,
    E: TokenEstimator + ?Sized,
{
    let mut rendered = String::new();
    messages
        .iter()
        .map(|message| {
            rendered.clear();
            message.render_for_compaction(&mut rendered);
            estimator.estimate(&rendered)
        })
        .sum()
}

/// Return true when compaction should run for this transcript.
pub fn should_compact<M, E>(messages: &[M], config: &CompactionConfig, estimator: &E) -> bool
where
    M: TranscriptMessage,
    E: TokenEstimator + ?Sized,
{
    config.enabled()
        && estimate_transcript_tokens(messages, estimator) >= config.auto_compact_token_limit
}

/// Build a compaction prompt plus deterministic finalization plan.
pub fn prepare_compaction<M, E>(
    messages: &[M],
    config: &CompactionConfig,
    estimator: &E,
) -> Option<PreparedCompaction>
where
    M: TranscriptMessage,
    E: TokenEstimator + ?Sized,
{
    let estimated_transcript_tokens = estimate_transcript_tokens(messages, estimator);
    if !config.enabled() || estimated_transcript_tokens < config.auto_compact_token_limit {
        return None;
    }

    let mut rendered_messages = Vec::new();
    let mut token_total = estimator.estimate(&config.compaction_prompt);
    let mut scratch = String::new();
    let mut omitted_messages = 0usize;

    for idx in (0..messages.len()).rev() {
        scratch.clear();
        messages[idx].render_for_compaction(&mut scratch);
        let tokens = estimator.estimate(&scratch);
        if token_total.saturating_add(tokens) > config.compact_request_token_limit
            && !rendered_messages.is_empty()
        {
            omitted_messages = idx + 1;
            break;
        }
        token_total = token_total.saturating_add(tokens);
        rendered_messages.push(scratch.clone());
    }
    rendered_messages.reverse();

    if rendered_messages.is_empty() {
        return None;
    }

    let mut prompt = String::new();
    prompt.push_str(&config.compaction_prompt);
    prompt.push_str("\n\nConversation transcript:\n\n");
    prompt.push_str(&rendered_messages.join("\n\n"));
    if omitted_messages > 0 {
        prompt.push_str("\n\nNote: ");
        prompt.push_str(&omitted_messages.to_string());
        prompt.push_str(
            " oldest message(s) were omitted because the compaction request was too large.",
        );
    }

    let plan = CompactionPlan {
        recent_user_messages: recent_user_messages(messages, config, estimator),
        summary_prefix: config.summary_prefix.clone(),
    };
    let estimated_request_tokens = estimator.estimate(&prompt);

    Some(PreparedCompaction {
        request: CompactionRequest {
            prompt,
            omitted_messages,
            estimated_transcript_tokens,
            estimated_request_tokens,
        },
        plan,
    })
}

/// Combine a model-produced summary with a previously prepared plan.
pub fn finalize_compaction(plan: &CompactionPlan, summary: &str) -> CompactedTranscript {
    let summary = summary.trim();
    let summary = if summary.is_empty() {
        "(no summary was produced)"
    } else {
        summary
    };

    let summary_message = if plan.summary_prefix.trim().is_empty() {
        summary.to_string()
    } else {
        format!("{}\n{}", plan.summary_prefix.trim_end(), summary)
    };

    CompactedTranscript {
        summary_message,
        recent_user_messages: plan.recent_user_messages.clone(),
    }
}

fn recent_user_messages<M, E>(
    messages: &[M],
    config: &CompactionConfig,
    estimator: &E,
) -> Vec<String>
where
    M: TranscriptMessage,
    E: TokenEstimator + ?Sized,
{
    if config.recent_user_token_budget == 0 {
        return Vec::new();
    }

    let mut selected = Vec::new();
    let mut remaining = config.recent_user_token_budget;
    let mut scratch = String::new();

    for message in messages.iter().rev() {
        scratch.clear();
        if !message.user_text_for_compaction(&mut scratch) || scratch.is_empty() {
            continue;
        }
        if message.is_compaction_summary(&config.summary_prefix) {
            continue;
        }

        let tokens = estimator.estimate(&scratch);
        if tokens <= remaining {
            selected.push(scratch.clone());
            remaining = remaining.saturating_sub(tokens);
            continue;
        }

        if remaining > 0 {
            selected.push(truncate_to_token_budget_with_estimator(
                &scratch, remaining, estimator,
            ));
        }
        break;
    }

    selected.reverse();
    selected
}
