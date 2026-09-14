// SPDX-License-Identifier: AGPL-3.0-only

//! Temporary limits for V4's eager, single-sequence compressor state.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;

use crate::cli::ServeArgs;

pub(crate) fn validate_v4_bringup(args: &ServeArgs, config: &ModelConfig) -> Result<()> {
    if config.model_type != "deepseek_v4" {
        return Ok(());
    }
    ensure!(
        !(args.speculative || args.self_speculative || args.ngram_speculative || args.dflash),
        "DeepSeek V4 bring-up requires speculation off; native DSpark is not implemented"
    );
    ensure!(
        args.max_batch_size == 1,
        "DeepSeek V4 bring-up requires --max-batch-size 1: compressor state is single-sequence"
    );
    ensure!(
        (1..=2048).contains(&args.max_seq_len),
        "DeepSeek V4 bring-up requires --max-seq-len in 1..=2048 (prompt plus output); sparse index selection is not implemented"
    );
    ensure!(
        !args.prefix_caching_enabled(),
        "DeepSeek V4 bring-up requires prefix caching off: KV alone cannot restore compressor state"
    );
    ensure!(
        !args.high_speed_swap && args.swap_space_gb == 0,
        "DeepSeek V4 bring-up requires --swap-space-gb 0 and high-speed swap off"
    );
    Ok(())
}

/// Validate the resolved budget, including automatic memory-budget adjustments.
/// The existing V4 prefill resets and seeds compressor state for a whole prompt;
/// continuing a partially processed prompt would reset it at the wrong offset.
pub(crate) fn validate_v4_prefill_budget(
    config: &ModelConfig,
    max_seq_len: usize,
    prefill_budget: usize,
) -> Result<()> {
    ensure!(
        config.model_type != "deepseek_v4" || prefill_budget >= max_seq_len,
        "DeepSeek V4 bring-up requires a prefill budget at least --max-seq-len; chunk continuation is not implemented"
    );
    Ok(())
}

#[cfg(test)]
#[path = "v4_bringup_tests.rs"]
mod tests;
