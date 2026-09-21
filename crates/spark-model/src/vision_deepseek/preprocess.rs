// SPDX-License-Identifier: AGPL-3.0-only

//! Decode → plan → resample → patchify, the pixel half of `load_image`.

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};

use super::geometry::{DsImagePlan, DsVisionGeometry, plan_image};
pub use super::resample::{pad_contain_gray, resize_pil_bicubic_rgb};

/// Decoder limit: reject a header declaring more than this on either side before a single
/// pixel is allocated.
///
/// ★ DUPLICATION, deliberate and flagged. `vision_preprocess.rs` owns the original
/// `decode_image` together with `DECODE_MAX_SIDE` / `DECODE_MAX_ALLOC`, but both the
/// function and the constants are private there and that file belongs to another unit. The
/// right fix is lifting `decode_image` to `pub(crate)` from a change that owns
/// `vision_preprocess.rs`; until then these two constants and [`decode_rgb8`] below are a
/// restatement, and must be kept in step with it.
const DECODE_MAX_SIDE: u32 = 16_384;

/// Decoder limit: bytes the decoder may hold at once for one image. See the note on
/// [`DECODE_MAX_SIDE`]; `image`'s own 512 MiB default is a per-request budget competing
/// with the KV cache on GB10's unified memory, from an unauthenticated HTTP body.
const DECODE_MAX_ALLOC: u64 = 192 * 1024 * 1024;

/// ViT patches for one image.
#[derive(Debug, Clone, PartialEq)]
pub struct DsImagePatches {
    /// `[n_vit_h · n_vit_w, 3 · p · p]`, patch-major with `(c, y, x)` inner order, in
    /// `[-1, 1]`. f32 here; the GPU path casts to bf16.
    pub pixels: Vec<f32>,
    pub plan: DsImagePlan,
}

/// SBIO: `uri` must already be a `data:` URI — the server resolves `http(s)` itself.
pub fn preprocess_image_deepseek(uri: &str, g: &DsVisionGeometry) -> Result<DsImagePatches> {
    if !uri.starts_with("data:") {
        bail!(
            "the DeepSeek image preprocessor performs no I/O: expected a `data:` URI, \
             got `{}`",
            uri.chars().take(24).collect::<String>()
        );
    }
    let img = decode_rgb8(uri)?;
    let (width, height) = (img.width() as usize, img.height() as usize);
    let plan = plan_image(width, height, g)?;
    let src = img.to_rgb8().into_raw();

    // The reference's two branches: a squashed image is plain-resized (the gray bars would
    // otherwise swallow most of the frame), everything else is pad-contained.
    let rgb = if plan.squash {
        resize_pil_bicubic_rgb(&src, width, height, plan.best_width, plan.best_height)
    } else {
        pad_contain_gray(&src, width, height, plan.best_width, plan.best_height)
    };

    let pixels = patches_from_rgb8(&rgb, &plan, g.patch_size);
    Ok(DsImagePatches { pixels, plan })
}

/// The reference's `reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4)` plus `(v/255 - .5)/.5`.
///
/// The reference casts to bf16 at this point; Atlas keeps f32 and lets the GPU path cast
/// (C4's `vision_f32_to_bf16`), so the host reference and the goldens stay exact.
pub fn patches_from_rgb8(rgb: &[u8], plan: &DsImagePlan, patch: usize) -> Vec<f32> {
    let (nh, nw, bw) = (plan.n_vit_h, plan.n_vit_w, plan.best_width);
    debug_assert_eq!(
        rgb.len(),
        bw * plan.best_height * 3,
        "resampled image does not match the plan"
    );
    let mut out = vec![0.0f32; nh * nw * 3 * patch * patch];
    for ph in 0..nh {
        for pw in 0..nw {
            for c in 0..3 {
                let base = ((ph * nw + pw) * 3 + c) * patch * patch;
                for y in 0..patch {
                    let row = (ph * patch + y) * bw;
                    for x in 0..patch {
                        let s = (row + pw * patch + x) * 3 + c;
                        out[base + y * patch + x] = (rgb[s] as f32 / 255.0 - 0.5) / 0.5;
                    }
                }
            }
        }
    }
    out
}

/// Decode a `data:` URI into an RGB-capable image, with the same hardening the Qwen path
/// applies. See the note on [`DECODE_MAX_SIDE`] for why this is restated here.
fn decode_rgb8(data_uri: &str) -> Result<DynamicImage> {
    let (_mime, bytes) = crate::vision_preprocess::decode_data_uri_bytes(data_uri)?;
    if bytes.is_empty() {
        bail!("image payload is empty");
    }
    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    let mut reader = ImageReader::new(std::io::Cursor::new(&bytes));
    reader.set_format(fmt);
    let mut limits = Limits::default();
    limits.max_image_width = Some(DECODE_MAX_SIDE);
    limits.max_image_height = Some(DECODE_MAX_SIDE);
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);

    // ★ DELIBERATE DIVERGENCE FROM THE REFERENCE. DeepSeek does a bare
    // `Image.open(...).convert("RGB")`, and PIL does NOT honour the EXIF orientation tag;
    // Atlas's Qwen path does, because a phone photo stored with `Orientation = 6` is
    // otherwise handed to the model a quarter turn over — it does not error, it just
    // answers about a sideways picture, and every viewer the user compares against honours
    // the tag. Applying it here keeps the two vision families consistent. It cannot move a
    // golden (all fixtures are synthetic PNGs, which carry no orientation), but it is a
    // real divergence from "exact reference parity" and belongs in the accepted-deviation
    // list.
    let mut decoder = reader.into_decoder().context("image decode failed")?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).context("image decode failed")?;
    if orientation != image::metadata::Orientation::NoTransforms {
        tracing::debug!("applying EXIF orientation {orientation:?}");
        img.apply_orientation(orientation);
    }
    if img.width() == 0 || img.height() == 0 {
        bail!("decoded image has a zero dimension");
    }
    Ok(img)
}

#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
