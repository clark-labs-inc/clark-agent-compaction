//! Safe formatting for readable private reasoning in compaction requests.

/// Default character cap for one private-reasoning excerpt.
pub const DEFAULT_PRIVATE_REASONING_EXCERPT_CHARS: usize = 2_000;

const HEADER: &str =
    "[private reasoning — non-user context; distill durable findings, do not quote]\n";
const TRUNCATION_MARKER: &str =
    "\n[private reasoning excerpt truncated; retain only durable findings]\n";

/// Append one bounded private-reasoning excerpt to a compaction transcript.
///
/// Callers must supply only readable, non-opaque reasoning text. This helper
/// deliberately does not know provider wire formats, so signed or encrypted
/// replay artifacts remain the caller's responsibility to exclude.
pub fn append_private_reasoning_for_compaction(
    out: &mut String,
    reasoning: &str,
    max_chars: usize,
) -> bool {
    let reasoning = reasoning.trim();
    if reasoning.is_empty() || max_chars == 0 {
        return false;
    }

    out.push_str(HEADER);
    out.push_str(&truncate_reasoning(reasoning, max_chars));
    out.push('\n');
    true
}

fn truncate_reasoning(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let marker_chars = TRUNCATION_MARKER.chars().count();
    if max_chars <= marker_chars {
        return text.chars().take(max_chars).collect();
    }

    let retained = max_chars - marker_chars;
    let head_chars = retained / 3;
    let tail_chars = retained.saturating_sub(head_chars);
    let head: String = text.chars().take(head_chars).collect();
    let tail: String = text
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}{TRUNCATION_MARKER}{tail}")
}
