# clark-agent-compaction

Small, provider-agnostic context compaction primitives for agent transcripts.

This crate plans a compaction request and finalizes the model's summary into a
handoff message plus retained recent user messages. It does not know about HTTP,
providers, API keys, env vars, persistence, or any specific agent runtime.

It is designed to:

- trigger compaction from an approximate token threshold
- ask a model for a concise handoff over the current transcript
- encode the summary as a model-visible user-shaped runtime context item
- keep recent real user requests so the active instruction survives
- filter earlier compaction summaries out of the retained user-message tail

Default limits are tuned for a large 1M-token window:

- compact when the rendered transcript is around `300_000` estimated tokens
- cap the summary request source at around `250_000` estimated tokens
- retain around `20_000` estimated tokens of recent user messages

Callers can and should override these per model.

## Minimal Flow

```rust
use clark_agent_compaction::{
    finalize_compaction, prepare_compaction, CharHeuristic, CompactionConfig,
    PlainMessage,
};

let transcript = vec![
    PlainMessage::user("Please migrate the code."),
    PlainMessage::assistant("I inspected the repo."),
];

let config = CompactionConfig::default();
let estimator = CharHeuristic;

if let Some(prepared) = prepare_compaction(&transcript, &config, &estimator) {
    // Send prepared.request.prompt through your own model/provider.
    let summary = "The user wants the migration finished. The repo has been inspected.";
    let compacted = finalize_compaction(&prepared.plan, summary);

    // Your runtime decides how to map these strings back into its typed message
    // format and where to place the summary relative to retained user messages.
    assert!(compacted.summary_message.contains("migration"));
}
```

For runtime-specific transcripts, implement `TranscriptMessage` instead of using
`PlainMessage`.
