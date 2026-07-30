//! Storage-neutral policy for append-only episodic compaction.
//!
//! A durable store measures the pressure in its own event log and retains its
//! own cutoff or watermark. This module decides whether that measured pressure
//! warrants a new episode summary, so every product follows the same policy
//! without coupling this crate to SQL, UUIDs, or a particular event schema.

/// Default policy for long-lived, append-only conversation histories.
///
/// The context-pressure threshold is model-specific and therefore disabled by
/// default. The unfinished-turn and age-based triggers provide safe fallback
/// bounds for conversations whose provider budget is not known at the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodicCompactionConfig {
    /// Trigger once the caller's provider-view token estimate reaches this
    /// threshold. `None` disables this model-aware trigger.
    pub trigger_context_tokens: Option<usize>,
    /// Trigger after this many user messages followed the most recent clean
    /// run completion in the current episode.
    pub trigger_unfinished_turns: usize,
    /// Require this conversation age, in whole days, for the long-tail trigger.
    pub trigger_age_days: u64,
    /// Require this many total user messages for the long-tail trigger.
    pub trigger_age_turn_count: usize,
}

impl Default for EpisodicCompactionConfig {
    fn default() -> Self {
        Self {
            trigger_context_tokens: None,
            trigger_unfinished_turns: 8,
            trigger_age_days: 30,
            trigger_age_turn_count: 50,
        }
    }
}

/// Durable reason an episodic boundary was written.
///
/// Stores can persist [`Self::as_str`] for diagnostics without learning any
/// policy details. `ManualRecovery` is never selected automatically; it is
/// reserved for an operator-owned clean-slate boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodicCompactionTrigger {
    ContextPressure,
    UnfinishedTurns,
    AgeAndTurns,
    ManualRecovery,
}

impl EpisodicCompactionTrigger {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextPressure => "context_pressure",
            Self::UnfinishedTurns => "unfinished_turns",
            Self::AgeAndTurns => "age_and_turns",
            Self::ManualRecovery => "manual_recovery",
        }
    }
}

/// Storage-neutral observations for one append-only conversation episode.
///
/// The store calculates these fields relative to its latest summary boundary.
/// It keeps any storage-specific pointer, such as an event sequence or record
/// ID, outside this type and writes it only after this policy selects a trigger.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpisodicCompactionPressure {
    /// Provider-view token estimate supplied by the runtime, if available.
    pub estimated_context_tokens: Option<usize>,
    /// User messages since the latest summary boundary. This is metadata for
    /// the eventual summary record and does not itself decide a trigger.
    pub user_messages_since_boundary: usize,
    /// Clean runs completed since the latest summary boundary.
    pub run_completed_since_boundary: usize,
    /// User messages after the latest clean run completion, or after the
    /// summary boundary when none completed.
    pub user_messages_after_latest_run_completed: usize,
    /// Conversation age in whole days.
    pub conversation_age_days: u64,
    /// Total user messages since conversation inception.
    pub total_user_messages: usize,
}

/// Select the highest-priority automatic episodic-compaction trigger.
///
/// Context pressure wins because the next provider request may overflow.
/// Unfinished turns then protect a conversation that has accumulated failed or
/// stalled attempts. The age-and-turn count rule bounds healthy long-running
/// conversations without compacting sparse older ones.
pub fn episodic_compaction_trigger(
    pressure: &EpisodicCompactionPressure,
    config: &EpisodicCompactionConfig,
) -> Option<EpisodicCompactionTrigger> {
    if let (Some(estimated), Some(trigger)) = (
        pressure.estimated_context_tokens,
        config.trigger_context_tokens,
    ) {
        if estimated >= trigger {
            return Some(EpisodicCompactionTrigger::ContextPressure);
        }
    }

    if pressure.user_messages_after_latest_run_completed >= config.trigger_unfinished_turns {
        return Some(EpisodicCompactionTrigger::UnfinishedTurns);
    }

    if pressure.conversation_age_days >= config.trigger_age_days
        && pressure.total_user_messages >= config.trigger_age_turn_count
    {
        return Some(EpisodicCompactionTrigger::AgeAndTurns);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_pressure_wins_before_a_provider_overflow() {
        let pressure = EpisodicCompactionPressure {
            estimated_context_tokens: Some(408_000),
            user_messages_since_boundary: 80,
            run_completed_since_boundary: 80,
            conversation_age_days: 1,
            total_user_messages: 80,
            ..EpisodicCompactionPressure::default()
        };
        let config = EpisodicCompactionConfig {
            trigger_context_tokens: Some(400_000),
            ..EpisodicCompactionConfig::default()
        };

        assert_eq!(
            episodic_compaction_trigger(&pressure, &config),
            Some(EpisodicCompactionTrigger::ContextPressure)
        );
    }

    #[test]
    fn unfinished_turns_only_count_after_the_latest_clean_completion() {
        let pressure = EpisodicCompactionPressure {
            user_messages_since_boundary: 20,
            run_completed_since_boundary: 1,
            user_messages_after_latest_run_completed: 9,
            conversation_age_days: 1,
            total_user_messages: 20,
            ..EpisodicCompactionPressure::default()
        };

        assert_eq!(
            episodic_compaction_trigger(&pressure, &EpisodicCompactionConfig::default()),
            Some(EpisodicCompactionTrigger::UnfinishedTurns)
        );
    }

    #[test]
    fn a_recent_clean_completion_prevents_the_unfinished_turn_trigger() {
        let pressure = EpisodicCompactionPressure {
            user_messages_since_boundary: 20,
            run_completed_since_boundary: 1,
            user_messages_after_latest_run_completed: 2,
            conversation_age_days: 1,
            total_user_messages: 20,
            ..EpisodicCompactionPressure::default()
        };

        assert_eq!(
            episodic_compaction_trigger(&pressure, &EpisodicCompactionConfig::default()),
            None
        );
    }

    #[test]
    fn context_pressure_is_disabled_until_the_runtime_supplies_a_limit() {
        let pressure = EpisodicCompactionPressure {
            estimated_context_tokens: Some(900_000),
            user_messages_since_boundary: 80,
            run_completed_since_boundary: 80,
            conversation_age_days: 1,
            total_user_messages: 80,
            ..EpisodicCompactionPressure::default()
        };

        assert_eq!(
            episodic_compaction_trigger(&pressure, &EpisodicCompactionConfig::default()),
            None
        );
    }

    #[test]
    fn age_trigger_requires_both_age_and_turn_count() {
        let sparse = EpisodicCompactionPressure {
            conversation_age_days: 60,
            total_user_messages: 10,
            ..EpisodicCompactionPressure::default()
        };
        assert_eq!(
            episodic_compaction_trigger(&sparse, &EpisodicCompactionConfig::default()),
            None
        );

        let old_busy = EpisodicCompactionPressure {
            conversation_age_days: 60,
            total_user_messages: 200,
            ..EpisodicCompactionPressure::default()
        };
        assert_eq!(
            episodic_compaction_trigger(&old_busy, &EpisodicCompactionConfig::default()),
            Some(EpisodicCompactionTrigger::AgeAndTurns)
        );
    }

    #[test]
    fn trigger_names_are_stable_for_store_metadata() {
        assert_eq!(
            EpisodicCompactionTrigger::ContextPressure.as_str(),
            "context_pressure"
        );
        assert_eq!(
            EpisodicCompactionTrigger::ManualRecovery.as_str(),
            "manual_recovery"
        );
    }
}
