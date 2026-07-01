/// Default runtime-context marker for model-visible compaction handoffs.
pub const DEFAULT_SUMMARY_PREFIX: &str = "[runtime context — compacted transcript handoff, not a new user instruction]\nA previous agent compacted the earlier part of this conversation. Use this handoff only as background, preserve the concrete current user request that follows it, and continue the session:";

/// Default prompt for the caller-owned summarization model request.
pub const DEFAULT_COMPACTION_PROMPT: &str = r#"Create a concise handoff summary for another coding agent that will continue this exact session.

Include:
- The current user request or active task in concrete terms
- Current progress and decisions already made
- Important constraints, user preferences, and safety rules
- Files, commands, errors, tool results, and facts needed to continue
- Clear next steps

Do not say the task is missing if the transcript contains a user request. Be specific, preserve concrete paths and identifiers, and omit filler."#;

/// Default estimated-token threshold for automatic compaction.
pub const DEFAULT_AUTO_COMPACT_TOKEN_LIMIT: usize = 300_000;
/// Default source budget for the compaction request itself.
pub const DEFAULT_COMPACT_REQUEST_TOKEN_LIMIT: usize = 250_000;
/// Default budget for recent real user messages retained after compaction.
pub const DEFAULT_RECENT_USER_TOKEN_BUDGET: usize = 20_000;

/// Approximate token estimator used by compaction budgets.
///
/// Compaction only needs a conservative threshold signal; exact provider
/// tokenizer coupling belongs in the caller. Custom estimators can override
/// `max_chars_for_tokens` to tune truncation behavior.
pub trait TokenEstimator {
    /// Estimate token count for `text`.
    fn estimate(&self, text: &str) -> usize;

    /// Convert a token budget into an approximate character budget.
    fn max_chars_for_tokens(&self, tokens: usize) -> usize {
        tokens.saturating_mul(4)
    }
}

/// Fast four-characters-per-token heuristic.
#[derive(Clone, Copy, Debug, Default)]
pub struct CharHeuristic;

impl TokenEstimator for CharHeuristic {
    fn estimate(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

/// Configuration for compaction planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionConfig {
    pub auto_compact_token_limit: usize,
    pub compact_request_token_limit: usize,
    pub recent_user_token_budget: usize,
    pub summary_prefix: String,
    pub compaction_prompt: String,
}

impl CompactionConfig {
    /// Build a config with custom numeric limits and default text.
    pub fn with_limits(
        auto_compact_token_limit: usize,
        compact_request_token_limit: usize,
        recent_user_token_budget: usize,
    ) -> Self {
        Self {
            auto_compact_token_limit,
            compact_request_token_limit,
            recent_user_token_budget,
            ..Self::default()
        }
    }

    /// Disable automatic compaction while preserving prompt defaults.
    pub fn disabled() -> Self {
        Self {
            auto_compact_token_limit: usize::MAX,
            compact_request_token_limit: usize::MAX,
            recent_user_token_budget: usize::MAX,
            ..Self::default()
        }
    }

    /// Whether automatic compaction is enabled.
    pub fn enabled(&self) -> bool {
        self.auto_compact_token_limit != usize::MAX
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            auto_compact_token_limit: DEFAULT_AUTO_COMPACT_TOKEN_LIMIT,
            compact_request_token_limit: DEFAULT_COMPACT_REQUEST_TOKEN_LIMIT,
            recent_user_token_budget: DEFAULT_RECENT_USER_TOKEN_BUDGET,
            summary_prefix: DEFAULT_SUMMARY_PREFIX.to_string(),
            compaction_prompt: DEFAULT_COMPACTION_PROMPT.to_string(),
        }
    }
}
