# clark-agent-compaction

Small, provider-agnostic context compaction primitives for agent transcripts.

This crate plans a compaction request and finalizes the model's summary into a
handoff message plus retained recent user messages. It does not perform model
calls or own persistence; callers provide transcript rendering, token
estimation, and model/provider I/O.

It pairs naturally with [`clark-agent`](https://github.com/clark-labs-inc/clark-agent),
Clark's typed agent loop crate.

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

## With `clark-agent`

```rust
use clark_agent as ca;
use clark_agent_compaction::{
    finalize_compaction, prepare_compaction, CharHeuristic, CompactionConfig,
    TranscriptMessage,
};

struct AgentMessageView<'a>(&'a ca::AgentMessage);

impl TranscriptMessage for AgentMessageView<'_> {
    fn render_for_compaction(&self, out: &mut String) {
        match self.0 {
            ca::AgentMessage::System { content, .. } => {
                out.push_str("[system]\n");
                out.push_str(content);
            }
            ca::AgentMessage::User { content, .. } => {
                out.push_str("[user]\n");
                render_user_content(content, out);
            }
            ca::AgentMessage::Assistant { content, .. } => {
                out.push_str("[assistant]\n");
                let text = content.plain_text();
                if text.is_empty() {
                    out.push_str("(empty)");
                } else {
                    out.push_str(&text);
                }
            }
            ca::AgentMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
                ..
            } => {
                let status = if *is_error { "error" } else { "ok" };
                out.push_str(&format!(
                    "[tool result {tool_call_id} {tool_name} {status}]\n{}",
                    content.plain_text()
                ));
            }
            ca::AgentMessage::Custom { kind, payload, .. } => {
                out.push_str(&format!("[custom {kind}]\n{payload}"));
            }
        }
    }

    fn user_text_for_compaction(&self, out: &mut String) -> bool {
        let ca::AgentMessage::User { content, .. } = self.0 else {
            return false;
        };
        render_user_text(content, out);
        true
    }
}

fn render_user_content(content: &ca::UserContent, out: &mut String) {
    match content {
        ca::UserContent::Text(text) => out.push_str(text),
        ca::UserContent::Blocks(blocks) => {
            for block in blocks {
                match block {
                    ca::UserBlock::Text(text) => out.push_str(&text.text),
                    ca::UserBlock::Image(image) => {
                        out.push_str(image.alt.as_deref().unwrap_or("[image]"));
                    }
                }
                out.push('\n');
            }
        }
    }
}

fn render_user_text(content: &ca::UserContent, out: &mut String) {
    match content {
        ca::UserContent::Text(text) => out.push_str(text),
        ca::UserContent::Blocks(blocks) => {
            for block in blocks {
                if let ca::UserBlock::Text(text) = block {
                    out.push_str(&text.text);
                    out.push('\n');
                }
            }
        }
    }
}

fn user_message(text: impl Into<String>) -> ca::AgentMessage {
    ca::AgentMessage::User {
        content: ca::UserContent::Text(text.into()),
        timestamp: None,
    }
}

async fn compact_with_clark_agent(
    messages: Vec<ca::AgentMessage>,
    config: &CompactionConfig,
) -> Vec<ca::AgentMessage> {
    let views = messages.iter().map(AgentMessageView).collect::<Vec<_>>();

    let Some(prepared) = prepare_compaction(&views, config, &CharHeuristic) else {
        return messages;
    };

    // Send prepared.request.prompt through the same model/provider path your
    // runtime already uses for summary turns.
    let summary = call_summary_model(&prepared.request.prompt).await;
    let compacted = finalize_compaction(&prepared.plan, &summary);

    let mut replacement = vec![user_message(compacted.summary_message)];
    replacement.extend(compacted.recent_user_messages.into_iter().map(user_message));
    replacement
}

async fn call_summary_model(prompt: &str) -> String {
    // Use your existing clark-agent StreamFn/provider integration here.
    todo!("send compaction prompt to the configured model: {prompt}")
}
```

In a `clark-agent` integration this usually lives inside a `ContextTransform`
plugin. `should_compact` decides whether to run, `prepare_compaction` builds the
summary request, and `finalize_compaction` gives you the replacement
model-visible history.

For tests or prototypes, the crate also includes `PlainMessage`.
