use crate::TranscriptMessage;

/// Simple test/example transcript shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlainMessage {
    System(String),
    User(String),
    Assistant {
        text: String,
        tool_calls: Vec<PlainToolCall>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
        is_error: bool,
    },
    Custom {
        kind: String,
        payload: String,
    },
}

impl PlainMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self::System(text.into())
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::User(text.into())
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant {
            text: text.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_with_tool_calls(
        text: impl Into<String>,
        tool_calls: Vec<PlainToolCall>,
    ) -> Self {
        Self::Assistant {
            text: text.into(),
            tool_calls,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: content.into(),
            is_error,
        }
    }
}

/// Tool-call rendering helper for [`PlainMessage`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlainToolCall {
    pub name: String,
    pub arguments: String,
}

impl PlainToolCall {
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

impl TranscriptMessage for PlainMessage {
    fn render_for_compaction(&self, out: &mut String) {
        match self {
            PlainMessage::System(text) => {
                out.push_str("[system]\n");
                out.push_str(text);
            }
            PlainMessage::User(text) => {
                out.push_str("[user]\n");
                out.push_str(text);
            }
            PlainMessage::Assistant { text, tool_calls } => {
                out.push_str("[assistant]\n");
                let mut wrote = false;
                if !text.is_empty() {
                    out.push_str(text);
                    wrote = true;
                }
                if !tool_calls.is_empty() {
                    if wrote {
                        out.push('\n');
                    }
                    out.push_str("tool calls: ");
                    for (idx, call) in tool_calls.iter().enumerate() {
                        if idx > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&call.name);
                        out.push('(');
                        out.push_str(&call.arguments);
                        out.push(')');
                    }
                    wrote = true;
                }
                if !wrote {
                    out.push_str("(empty)");
                }
            }
            PlainMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
            } => {
                let status = if *is_error { "error" } else { "ok" };
                out.push_str("[tool result ");
                out.push_str(tool_call_id);
                out.push(' ');
                out.push_str(tool_name);
                out.push(' ');
                out.push_str(status);
                out.push_str("]\n");
                out.push_str(content);
            }
            PlainMessage::Custom { kind, payload } => {
                out.push_str("[custom ");
                out.push_str(kind);
                out.push_str("]\n");
                out.push_str(payload);
            }
        }
    }

    fn user_text_for_compaction(&self, out: &mut String) -> bool {
        if let PlainMessage::User(text) = self {
            out.push_str(text);
            true
        } else {
            false
        }
    }
}
