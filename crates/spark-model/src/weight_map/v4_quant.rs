// SPDX-License-Identifier: AGPL-3.0-only

//! On-disk packed V4 expert contract shared by preflight and loading.
//! Resolve each tensor group; global `quant_method: fp8` is not authoritative
//! for the routed experts in NVIDIA's mixed-precision checkpoints.

use anyhow::{Context, Result, ensure};
use spark_runtime::weights::{WeightDtype, WeightStore};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackedExpertFormat {
    Nvfp4,
    Mxfp4,
}

pub(crate) fn resolve_packed_expert(
    store: &WeightStore,
    prefix: &str,
) -> Result<PackedExpertFormat> {
    let name = format!("{prefix}.weight");
    let weight = store
        .get(&name)
        .with_context(|| format!("missing `{name}`"))?;
    ensure!(
        weight.dtype == WeightDtype::UInt8 && weight.shape.len() == 2,
        "`{name}` must be a packed U8/I8 matrix, got {:?} {:?}",
        weight.dtype,
        weight.shape
    );
    let n = weight.shape[0];
    let k = weight.shape[1]
        .checked_mul(2)
        .context("packed expert width overflow")?;
    ensure!(n > 0 && k > 0, "`{name}` must be nonempty");
    let nvfp4 = store.contains(&format!("{prefix}.weight_scale"))
        || store.contains(&format!("{prefix}.weight_scale_2"))
        || store.contains(&format!("{prefix}.input_scale"));
    if nvfp4 {
        ensure!(
            !store.contains(&format!("{prefix}.scale")),
            "`{prefix}` mixes NVFP4 and MXFP4 scale names"
        );
        ensure!(
            k.is_multiple_of(16),
            "`{prefix}` NVFP4 width must be divisible by 16"
        );
        expect(
            store,
            &format!("{prefix}.weight_scale"),
            WeightDtype::FP8E4M3,
            &[n, k / 16],
        )?;
        for suffix in ["weight_scale_2", "input_scale"] {
            let name = format!("{prefix}.{suffix}");
            let scalar = store
                .get(&name)
                .with_context(|| format!("missing `{name}`"))?;
            ensure!(
                scalar.dtype == WeightDtype::FP32
                    && (scalar.shape.is_empty() || scalar.shape == [1]),
                "`{name}` must be a scalar F32 tensor, got {:?} {:?}",
                scalar.dtype,
                scalar.shape
            );
        }
        Ok(PackedExpertFormat::Nvfp4)
    } else {
        ensure!(
            k.is_multiple_of(32),
            "`{prefix}` MXFP4 width must be divisible by 32"
        );
        expect(
            store,
            &format!("{prefix}.scale"),
            WeightDtype::FP8E8M0,
            &[n, k / 32],
        )?;
        Ok(PackedExpertFormat::Mxfp4)
    }
}

fn expect(store: &WeightStore, name: &str, dtype: WeightDtype, shape: &[usize]) -> Result<()> {
    let tensor = store
        .get(name)
        .with_context(|| format!("missing `{name}`"))?;
    ensure!(
        tensor.dtype == dtype && tensor.shape == shape,
        "`{name}` has {:?} {:?}, expected {dtype:?} {shape:?}",
        tensor.dtype,
        tensor.shape
    );
    Ok(())
}
