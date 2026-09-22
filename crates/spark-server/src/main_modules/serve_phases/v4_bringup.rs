// SPDX-License-Identifier: AGPL-3.0-only

//! Temporary limits for V4's eager, single-sequence compressor state.

use anyhow::{Result, ensure};
use avarok_core::config::ModelConfig;

use crate::cli::ServeArgs;

pub(crate) fn validate_v4_bringup(args: &ServeArgs, config: &ModelConfig) -> Result<()> {
    if config.model_type != "deepseek_v4" {
        return Ok(());
    }
    // `--speculative` is admitted ONLY on a checkpoint that declares native
    // DSpark (`dspark_block_size > 0`, i.e. the 0731 / Vision-Exp family). The
    // proposer that serves it is `DeepseekV4DsparkHead` (U3.1); the stage
    // contract itself is checked against the tensors in
    // `spark_model::preflight::deepseek_v4`, which is the layer that can see
    // them. Every OTHER speculation mode stays refused on V4: self-speculation,
    // n-gram and DFlash all drive the target's own verify ladder with state
    // this bring-up's eager compressor cannot restore.
    ensure!(
        !args.speculative || config.dspark_block_size > 0,
        "DeepSeek V4 bring-up requires speculation off on a checkpoint without native DSpark \
         (`dspark_block_size == 0`); the legacy single-module MTP is a different architecture"
    );
    ensure!(
        !(args.self_speculative || args.ngram_speculative || args.dflash),
        "DeepSeek V4 bring-up requires self-speculation, n-gram speculation and DFlash off; \
         native DSpark (`--speculative`) is the only supported drafter"
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
