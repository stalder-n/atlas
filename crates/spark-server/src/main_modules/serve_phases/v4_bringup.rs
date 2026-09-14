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
mod tests {
    use super::*;
    use clap::Parser;

    fn config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "deepseek_v4".into();
        config.compress_ratios = vec![0, 0, 4, 128];
        config
    }

    fn args() -> ServeArgs {
        ServeArgs::parse_from([
            "spark",
            "--max-batch-size",
            "1",
            "--max-seq-len",
            "2048",
            "--swap-space-gb",
            "0",
        ])
    }

    #[test]
    fn constrained_target_and_old_v4_configs_are_accepted() {
        let mut config = config();
        assert!(validate_v4_bringup(&args(), &config).is_ok());
        config.compress_ratios = vec![0, 4, 128];
        assert!(validate_v4_bringup(&args(), &config).is_ok());
    }

    #[test]
    fn unsafe_execution_modes_are_refused() {
        let config = config();
        let mutations: [fn(&mut ServeArgs); 10] = [
            |a: &mut ServeArgs| a.max_batch_size = 2,
            |a: &mut ServeArgs| a.max_seq_len = 2049,
            |a: &mut ServeArgs| a.max_seq_len = 0,
            |a: &mut ServeArgs| a.speculative = true,
            |a: &mut ServeArgs| a.self_speculative = true,
            |a: &mut ServeArgs| a.ngram_speculative = true,
            |a: &mut ServeArgs| a.dflash = true,
            |a: &mut ServeArgs| a.enable_prefix_caching = true,
            |a: &mut ServeArgs| a.swap_space_gb = 1,
            |a: &mut ServeArgs| a.high_speed_swap = true,
        ];
        for mutate in mutations {
            let mut args = args();
            mutate(&mut args);
            assert!(validate_v4_bringup(&args, &config).is_err());
        }
    }

    #[test]
    fn chunk_continuation_is_refused() {
        assert!(validate_v4_prefill_budget(&config(), 2048, 2048).is_ok());
        assert!(validate_v4_prefill_budget(&config(), 2048, 1024).is_err());
    }

    #[test]
    fn other_models_keep_existing_capabilities() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(validate_v4_bringup(&ServeArgs::parse_from(["spark"]), &config).is_ok());
        assert!(validate_v4_prefill_budget(&config, 32768, 1024).is_ok());
    }
}
