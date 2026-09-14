// SPDX-License-Identifier: AGPL-3.0-only

/// V4 bring-up counts emitted output as well as prompt tokens. The scheduler's
/// KV ceiling alone excludes the last sampled token (not yet inserted into KV).
pub(super) fn fits_context(
    is_v4: bool,
    max_seq_len: usize,
    prompt_len: usize,
    requested: usize,
) -> bool {
    !is_v4
        || prompt_len
            .checked_add(requested)
            .is_some_and(|total| total <= max_seq_len)
}

#[cfg(test)]
mod tests {
    use super::fits_context;

    #[test]
    fn v4_budget_includes_the_last_sampled_token() {
        assert!(fits_context(true, 2048, 2047, 1));
        assert!(!fits_context(true, 2048, 2047, 2));
        assert!(fits_context(true, 2048, 1024, 1024));
        assert!(!fits_context(true, 2048, usize::MAX, 1));
    }

    #[test]
    fn other_model_budgets_are_unchanged() {
        assert!(fits_context(false, 2048, 2047, 512));
    }
}
