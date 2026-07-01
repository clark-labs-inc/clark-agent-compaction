use clark_agent_compaction::{
    finalize_compaction, prepare_compaction, should_compact, truncate_to_token_budget,
    CharHeuristic, CompactionConfig, PlainMessage, PlainToolCall, DEFAULT_SUMMARY_PREFIX,
};

fn config() -> CompactionConfig {
    CompactionConfig::with_limits(20, 1_000, 12)
}

#[test]
fn does_not_prepare_below_threshold() {
    let transcript = vec![PlainMessage::user("short")];

    assert!(prepare_compaction(&transcript, &config(), &CharHeuristic).is_none());
}

#[test]
fn disabled_config_never_compacts() {
    let transcript = vec![PlainMessage::user("x ".repeat(1_000))];

    assert!(!should_compact(
        &transcript,
        &CompactionConfig::disabled(),
        &CharHeuristic
    ));
}

#[test]
fn request_renders_tool_context() {
    let transcript = vec![
        PlainMessage::user("please inspect the project"),
        PlainMessage::assistant_with_tool_calls(
            "reading",
            vec![PlainToolCall::new("read_file", r#"{"path":"src/main.rs"}"#)],
        ),
        PlainMessage::tool_result("call_1", "read_file", "fn main() {}", false),
    ];

    let prepared =
        prepare_compaction(&transcript, &config(), &CharHeuristic).expect("compaction request");

    assert!(prepared
        .request
        .prompt
        .contains(r#"read_file({"path":"src/main.rs"})"#));
    assert!(prepared
        .request
        .prompt
        .contains("[tool result call_1 read_file ok]"));
}

#[test]
fn request_omits_oldest_contiguous_prefix_when_budget_is_full() {
    let transcript = vec![
        PlainMessage::user("old ".repeat(200)),
        PlainMessage::assistant("middle ".repeat(200)),
        PlainMessage::user("new ".repeat(100)),
    ];
    let cfg = CompactionConfig::with_limits(1, 120, 1_000);

    let prepared =
        prepare_compaction(&transcript, &cfg, &CharHeuristic).expect("compaction request");

    assert_eq!(prepared.request.omitted_messages, 2);
    assert!(!prepared.request.prompt.contains("[assistant]\nmiddle"));
    assert!(prepared.request.prompt.contains("[user]\nnew"));
}

#[test]
fn oversized_single_latest_message_is_still_included() {
    let transcript = vec![PlainMessage::user("new ".repeat(1_000))];
    let cfg = CompactionConfig::with_limits(1, 2, 1_000);

    let prepared =
        prepare_compaction(&transcript, &cfg, &CharHeuristic).expect("compaction request");

    assert_eq!(prepared.request.omitted_messages, 0);
    assert!(prepared.request.prompt.contains("[user]\nnew"));
}

#[test]
fn finalization_keeps_recent_users_and_summary_separate() {
    let long_user = "important ".repeat(40);
    let transcript = vec![
        PlainMessage::user("old user message"),
        PlainMessage::assistant("assistant detail"),
        PlainMessage::tool_result("call_1", "read_file", "fn main() {}", false),
        PlainMessage::user(long_user),
    ];

    let prepared =
        prepare_compaction(&transcript, &config(), &CharHeuristic).expect("compaction request");
    let compacted = finalize_compaction(&prepared.plan, "summary");

    assert!(compacted
        .summary_message
        .starts_with(DEFAULT_SUMMARY_PREFIX));
    assert!(compacted.summary_message.contains("summary"));
    assert!(compacted
        .recent_user_messages
        .iter()
        .any(|text| text.contains("truncated during context compaction")));
}

#[test]
fn recent_user_retention_skips_prior_compaction_summaries() {
    let prior_summary = format!("{DEFAULT_SUMMARY_PREFIX}\nold summary");
    let transcript = vec![
        PlainMessage::user(prior_summary),
        PlainMessage::user("real request"),
    ];
    let cfg = CompactionConfig::with_limits(1, 1_000, 1_000);

    let prepared =
        prepare_compaction(&transcript, &cfg, &CharHeuristic).expect("compaction request");

    assert_eq!(prepared.plan.recent_user_messages, vec!["real request"]);
}

#[test]
fn truncating_recent_user_message_preserves_tail_instruction() {
    let text = format!(
        "{}\nNow answer with CLARK_LIVE_COMPACTION_DONE_3003.",
        "Important project context. ".repeat(900)
    );

    let truncated = truncate_to_token_budget(&text, 400);

    assert!(truncated.contains("[truncated during context compaction]"));
    assert!(truncated.contains("Now answer with CLARK_LIVE_COMPACTION_DONE_3003."));
}

#[test]
fn truncation_respects_utf8_boundaries() {
    let text = format!(
        "{}\nFinal instruction",
        "context with emoji \u{1F9EA} ".repeat(200)
    );

    let truncated = truncate_to_token_budget(&text, 80);

    assert!(truncated.is_char_boundary(truncated.len()));
    assert!(truncated.contains("Final instruction"));
}
