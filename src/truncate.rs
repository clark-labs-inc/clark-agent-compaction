use crate::{CharHeuristic, TokenEstimator};

/// Truncate text by the default character heuristic while preserving the tail.
pub fn truncate_to_token_budget(text: &str, budget: usize) -> String {
    truncate_to_token_budget_with_estimator(text, budget, &CharHeuristic)
}

/// Truncate text according to `estimator.max_chars_for_tokens`, preserving the tail.
pub fn truncate_to_token_budget_with_estimator<E>(
    text: &str,
    budget: usize,
    estimator: &E,
) -> String
where
    E: TokenEstimator + ?Sized,
{
    let max_chars = estimator.max_chars_for_tokens(budget);
    if text.len() <= max_chars {
        return text.to_string();
    }

    let marker = "\n[truncated during context compaction]\n";
    if max_chars <= marker.len() + 32 {
        let start = char_boundary_at_or_after(text, text.len().saturating_sub(max_chars));
        return format!("{marker}{}", text[start..].trim_start());
    }

    let keep_chars = max_chars.saturating_sub(marker.len());
    let head_chars = keep_chars / 2;
    let tail_chars = keep_chars.saturating_sub(head_chars);
    let head_end = char_boundary_at_or_before(text, head_chars);
    let tail_start = char_boundary_at_or_after(text, text.len().saturating_sub(tail_chars));

    format!(
        "{}{marker}{}",
        text[..head_end].trim_end(),
        text[tail_start..].trim_start()
    )
}

fn char_boundary_at_or_before(text: &str, idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }

    let mut out = 0;
    for (pos, _) in text.char_indices() {
        if pos > idx {
            break;
        }
        out = pos;
    }
    out
}

fn char_boundary_at_or_after(text: &str, idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }

    for (pos, _) in text.char_indices() {
        if pos >= idx {
            return pos;
        }
    }
    text.len()
}
