// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash-Vision-Exp tower + aligner + image-parameter loading.
//!
//! Tensor names are the checkpoint's own flat ones — `vision.*`, `aligner.*`,
//! `image_{start,pad,newline,end}` — the same set
//! [`crate::vision_deepseek_ref::load_weights_from_safetensors`] reads for the
//! host reference, so the two cannot drift.
//!
//! Every tensor goes through [`dense_auto`], not `dense`: the reference
//! declares `RMSNorm.weight` as `torch.float32`, and the FP8/NVFP4 repacks
//! quantize the projections. `dense_auto` passes BF16 through untouched,
//! truncates F32 → BF16 on device, and dequantizes FP8/NVFP4 — so a norm
//! weight is never read as garbage BF16.
//!
//! No rank logic here. The rank gate lives in `factory/build.rs`, which asks
//! for the tower only on rank 0 and for the image parameters on both.

use anyhow::{Context, Result, ensure};
use avarok_core::config::{ModelConfig, VisionConfig, VisionFlavor};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::{DeepseekVisionEncoder, DsImageParams, DsViTBlock};
use crate::weight_map::dense_auto;

/// The DeepSeek vision half of `config.vision`, or `None` for a text
/// checkpoint or a Qwen tower.
fn ds_vision(config: &ModelConfig) -> Option<&VisionConfig> {
    config
        .vision
        .as_ref()
        .filter(|v| v.flavor == VisionFlavor::DeepseekV4)
}

/// One tensor → a BF16 device pointer, naming it when it is missing (PCND: a
/// declared tower whose weights are absent must fail the load, not serve).
fn t(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<DevicePtr> {
    dense_auto(store, name, gpu)
        .with_context(|| {
            format!(
                "DeepSeek-V4 vision: `{name}` is missing or unreadable, but config.json \
                 declares a vision tower (`vision_n_layers > 0`)"
            )
        })
        .map(|w| w.weight)
}

/// Assert a tensor's on-disk shape, so a checkpoint whose tower differs from
/// what `config.json` declares fails at load with the two numbers in hand
/// rather than as silent garbage rows.
fn check_shape(store: &WeightStore, name: &str, want: &[usize]) -> Result<()> {
    let got = &store.get(name)?.shape;
    ensure!(
        got.as_slice() == want,
        "DeepSeek-V4 vision: `{name}` has shape {got:?}, expected {want:?} from config.json"
    );
    Ok(())
}

/// Bind the ViT tower + aligner. `Ok(None)` for a non-DeepSeek-vision config.
pub(super) fn load_vision_tower(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<DeepseekVisionEncoder>> {
    let Some(v) = ds_vision(config) else {
        return Ok(None);
    };
    let (dim, inter) = (v.hidden_size, v.intermediate_size);
    let patch_in = 3 * v.patch_size * v.patch_size;
    let unshuffled = dim * v.spatial_merge_size * v.spatial_merge_size;

    check_shape(store, "vision.patch_embed.proj.weight", &[dim, patch_in])?;
    let patch_w = t(store, gpu, "vision.patch_embed.proj.weight")?;
    let patch_b = t(store, gpu, "vision.patch_embed.proj.bias")?;

    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let b = format!("vision.blocks.{i}");
        // `w1` is the FUSED gate‖up projection (`2 · inter` rows); a tower that
        // ships them split would silently halve the MLP.
        check_shape(store, &format!("{b}.mlp.w1.weight"), &[2 * inter, dim])?;
        blocks.push(DsViTBlock {
            norm1_w: t(store, gpu, &format!("{b}.norm1.weight"))?,
            qkv_w: t(store, gpu, &format!("{b}.attn.wqkv.weight"))?,
            qkv_b: t(store, gpu, &format!("{b}.attn.wqkv.bias"))?,
            wo_w: t(store, gpu, &format!("{b}.attn.wo.weight"))?,
            wo_b: t(store, gpu, &format!("{b}.attn.wo.bias"))?,
            norm2_w: t(store, gpu, &format!("{b}.norm2.weight"))?,
            w1_w: t(store, gpu, &format!("{b}.mlp.w1.weight"))?,
            w2_w: t(store, gpu, &format!("{b}.mlp.w2.weight"))?,
        });
    }

    let norm_w = t(store, gpu, "vision.norm.weight")?;
    check_shape(store, "aligner.w1.weight", &[v.out_hidden_size, unshuffled])?;
    let enc = DeepseekVisionEncoder::new(
        patch_w,
        patch_b,
        blocks,
        norm_w,
        t(store, gpu, "aligner.w1.weight")?,
        t(store, gpu, "aligner.w1.bias")?,
        t(store, gpu, "aligner.w2.weight")?,
        t(store, gpu, "aligner.w2.bias")?,
        v,
        config.hidden_size,
        gpu,
    )?;
    tracing::info!(
        "DeepSeek-V4 vision tower bound: {} blocks, dim {dim}, aligner -> {}",
        v.depth,
        config.hidden_size
    );
    Ok(Some(enc))
}

/// Bind the four learned sentinel vectors. `Ok(None)` for a non-DeepSeek
/// vision config.
pub(super) fn load_image_params(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<DsImageParams>> {
    if ds_vision(config).is_none() {
        return Ok(None);
    }
    let h = config.hidden_size;
    for name in ["image_start", "image_pad", "image_newline", "image_end"] {
        check_shape(store, name, &[h])?;
    }
    Ok(Some(DsImageParams {
        start: t(store, gpu, "image_start")?,
        pad: t(store, gpu, "image_pad")?,
        newline: t(store, gpu, "image_newline")?,
        end: t(store, gpu, "image_end")?,
    }))
}
