// SPDX-License-Identifier: AGPL-3.0-only

//! The sizing half of `inference/image_processor.py`: `grid_tokens`, `solve_resize_ratio`,
//! `safe_resize`, and the `load_image` prologue that drives them.
//!
//! Every function below is a line-for-line port of the reference at revision
//! `86f746b36186f0e567729a5c06a8c918caba82a9`, with the reference line quoted in a comment
//! wherever the Python semantics (float `math.floor`, `int()` truncation, operator
//! precedence) are load-bearing. `fixtures/geometry.json` is the reference's own output.

use anyhow::{Result, bail};
use avarok_core::config::{VisionConfig, VisionFlavor};

/// `COMPRESS_PAD_TO` in the reference: the compressor's KV window, and therefore the
/// alignment every image block's START is padded onto.
pub const COMPRESS_PAD_TO: usize = 4;

/// Floor on `max_n_token`. Below this no grid can fit even a single aligner row plus the
/// START/END pair and the compress pad, so a config carrying less is a config bug, not a
/// tiny image budget (PCND — fail fast rather than silently resize to nothing).
const MIN_N_TOKEN: usize = 16;

/// Lowest `budget` the [`safe_resize`] loop may still call [`solve_resize_ratio`] with.
///
/// The reference's `while num_tokens > max_n_token` loop has no termination proof; this is
/// host code reachable from an unauthenticated HTTP body, so non-convergence must be a
/// bounded error rather than a spin. Expressed as a floor on the budget (not an iteration
/// count) so it says what it protects: below this every branch of `solve_resize_ratio`
/// degenerates. Unobservable on every golden case — see `geometry_tests.rs`.
const MIN_SOLVE_BUDGET: usize = 9;

/// The five `vision_*` config values the image pipeline needs, validated once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsVisionGeometry {
    pub patch_size: usize,
    pub downsample_ratio: usize,
    pub max_n_token: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: usize,
}

impl DsVisionGeometry {
    /// The primary constructor. Rejects degenerate values (PCND — no silent defaults).
    ///
    /// `min_pixels == 0` and `max_wh_ratio == 0` are the documented "no upscale" / "no
    /// clamp" sentinels (the reference's `None`), matching C1's Qwen mapping — they are
    /// values, not omissions, so they stay valid.
    pub fn new(
        patch_size: usize,
        downsample_ratio: usize,
        max_n_token: usize,
        min_pixels: usize,
        max_wh_ratio: usize,
    ) -> Result<Self> {
        if patch_size == 0 {
            bail!("vision_patch_size must be > 0");
        }
        if downsample_ratio == 0 {
            bail!("vision_downsample_ratio must be > 0");
        }
        if max_n_token < MIN_N_TOKEN {
            bail!("vision_max_n_token must be >= {MIN_N_TOKEN}, got {max_n_token}");
        }
        Ok(Self {
            patch_size,
            downsample_ratio,
            max_n_token,
            min_pixels,
            max_wh_ratio,
        })
    }

    /// Adapter over the config SSOT. The flavor must be `DeepseekV4`.
    pub fn from_vision_config(v: &VisionConfig) -> Result<Self> {
        if v.flavor != VisionFlavor::DeepseekV4 {
            bail!(
                "the DeepSeek image pipeline requires a DeepseekV4 vision tower, got {:?}",
                v.flavor
            );
        }
        // C1: the DeepSeek parser maps `vision_downsample_ratio` onto `spatial_merge_size`.
        Self::new(
            v.patch_size,
            v.spatial_merge_size,
            v.max_n_token,
            v.min_pixels,
            v.max_wh_ratio,
        )
    }
}

/// Everything `load_image`'s sizing half decides for one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsImagePlan {
    pub best_width: usize,
    pub best_height: usize,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    /// True ⇔ the ORIGINAL width ≥ `max_wh_ratio · height`, i.e. the reference takes the
    /// plain `image.resize(...)` branch instead of `ImageOps.pad`.
    pub squash: bool,
}

/// `n_vit / downsample_ratio`, rounded up — the reference's `math.ceil(n_vit / ds)`.
///
/// `ds == 0` is rejected by [`DsVisionGeometry::new`], so production can never reach the
/// division by zero; the debug assertion says so for anyone calling this directly.
pub fn n_llm(n_vit: usize, ds: usize) -> usize {
    debug_assert!(ds > 0, "downsample_ratio must be > 0");
    n_vit.div_ceil(ds)
}

/// `(n_llm_h, n_llm_w, num_tokens)` — the reference's `grid_tokens`.
pub fn grid_tokens(best_h: usize, best_w: usize, patch: usize, ds: usize) -> (usize, usize, usize) {
    debug_assert!(patch > 0, "patch_size must be > 0");
    // `math.ceil((best_height // patch_size) / downsample_ratio)` — the inner division is
    // Python's FLOOR division, the outer a ceiling.
    let n_llm_h = n_llm(best_h / patch, ds);
    let n_llm_w = n_llm(best_w / patch, ds);
    // `num_tokens = n_llm_h * (n_llm_w + 1) + 2` — one NEWLINE per grid row, plus START/END.
    let mut num_tokens = n_llm_h * (n_llm_w + 1) + 2;
    // An odd row count is padded up to an even one, because the compressor pairs rows.
    if n_llm_h % 2 == 1 {
        num_tokens += n_llm_w + 1;
    }
    // `num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2`. Python's `//`, `*` and `%`
    // share one precedence level and associate left to right, so this reads
    // `((((h + 1) // 2) * (w + 1)) % 2) * 2` — the pad that keeps the block's body an even
    // number of tokens. `(h + 1) // 2` on a non-negative int is `h.div_ceil(2)`, spelt that
    // way because clippy rejects the literal transcription.
    num_tokens += (n_llm_h.div_ceil(2) * (n_llm_w + 1)) % 2 * 2;
    (n_llm_h, n_llm_w, num_tokens)
}

/// `(n_llm_h, n_llm_w, best_height, best_width, num_tokens)` — the reference's
/// `solve_resize_ratio`. Its `assert max_w > 1` becomes an `Err`.
pub fn solve_resize_ratio(
    h: usize,
    w: usize,
    patch: usize,
    ds: usize,
    max_n_token: usize,
) -> Result<(usize, usize, usize, usize, usize)> {
    if h == 0 || w == 0 {
        bail!("solve_resize_ratio needs a non-empty image, got {w}x{h}");
    }
    if max_n_token < 3 {
        bail!("solve_resize_ratio needs a token budget of at least 3, got {max_n_token}");
    }
    // All of `r`, `max_w_float`, `max_h_float` and `beta` are f64, exactly as Python.
    let r = h as f64 / w as f64;
    let max_w_float = ((max_n_token as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;

    let (best_height, best_width) = if max_w_float < 1.0 {
        // Branch 1: a single aligner column; height takes the whole budget.
        let max_w: usize = 1;
        let mut max_h = (max_n_token - 2) / (max_w + 1);
        if max_h % 2 == 1 {
            max_h -= 1;
        }
        (max_h * patch * ds, max_w * patch * ds)
    } else if max_h_float < 2.0 {
        // Branch 2: two aligner rows; width takes the whole budget. The reference asserts
        // `max_w > 1` here — an `Err`, never a panic and never a degenerate grid.
        let max_h: usize = 2;
        let max_w = ((max_n_token - 2) / max_h) as i64 - 1;
        if max_w <= 1 {
            bail!(
                "solve_resize_ratio({h}, {w}, {patch}, {ds}, {max_n_token}): \
                 the reference asserts max_w > 1, got {max_w}"
            );
        }
        let max_w = max_w as usize;
        (max_h * patch * ds, max_w * patch * ds)
    } else {
        // Branch 3: both dimensions fit; scale by the tighter of the two ratios.
        let max_w = max_w_float.floor();
        let mut max_h = max_h_float.floor();
        if (max_h as i64) % 2 == 1 {
            max_h -= 1.0;
        }
        let pd = (patch * ds) as f64;
        let beta = (max_w * pd / w as f64).min(max_h * pd / h as f64);
        // `math.floor(width * beta / patch_size) * patch_size`
        let best_width = (w as f64 * beta / patch as f64).floor().max(0.0) as usize * patch;
        let best_height = (h as f64 * beta / patch as f64).floor().max(0.0) as usize * patch;
        (best_height, best_width)
    };

    let (n_llm_h, n_llm_w, num_tokens) = grid_tokens(best_height, best_width, patch, ds);
    Ok((n_llm_h, n_llm_w, best_height, best_width, num_tokens))
}

/// `(n_llm_h, n_llm_w, best_height, best_width)` — the reference's `safe_resize`.
pub fn safe_resize(
    h: usize,
    w: usize,
    best_h: usize,
    best_w: usize,
    patch: usize,
    ds: usize,
    max_n_token: usize,
) -> Result<(usize, usize, usize, usize)> {
    // `max_n_token -= COMPRESS_PAD_TO - 1`: the block's START may sit up to three tokens
    // into the compressor's window, and that pad has to fit in the same budget.
    let max_n_token = max_n_token.saturating_sub(COMPRESS_PAD_TO - 1);
    let (mut n_llm_h, mut n_llm_w, mut num_tokens) = grid_tokens(best_h, best_w, patch, ds);
    let (mut best_height, mut best_width) = (best_h, best_w);
    let mut budget = max_n_token;
    while num_tokens > max_n_token {
        // ★ ADDITION over the reference (see MIN_SOLVE_BUDGET): a bounded give-up.
        if budget < MIN_SOLVE_BUDGET {
            bail!(
                "safe_resize budget loop failed to converge for {w}x{h} \
                 (best {best_w}x{best_h}, budget fell to {budget}, target {max_n_token})"
            );
        }
        (n_llm_h, n_llm_w, best_height, best_width, num_tokens) =
            solve_resize_ratio(h, w, patch, ds, budget)?;
        budget -= 1;
    }
    Ok((n_llm_h, n_llm_w, best_height, best_width))
}

/// The sizing half of `load_image`. NOTE the argument order is `(width, height)` while every
/// reference helper takes `(height, width)`; the swap happens once, here.
pub fn plan_image(width: usize, height: usize, g: &DsVisionGeometry) -> Result<DsImagePlan> {
    if width == 0 || height == 0 {
        bail!("image has a zero dimension: {width}x{height}");
    }
    let (orig_width, orig_height) = (width, height);
    let (mut width, mut height) = (width, height);
    let p = g.patch_size;

    // `if args.vision_max_wh_ratio is not None and width > height * ratio: width = height * ratio`
    // — 0 is C1's "no clamp" sentinel, the reference's `None`.
    if g.max_wh_ratio != 0 && width > height * g.max_wh_ratio {
        width = height * g.max_wh_ratio;
    }
    // `if 0 < width * height < args.vision_min_pixels:` — upscale to the area floor. `int()`
    // on a positive float truncates, which is what `as usize` does.
    if g.min_pixels != 0 && width * height < g.min_pixels {
        let ratio = (g.min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let best_width = width.div_ceil(p) * p;
    let best_height = height.div_ceil(p) * p;

    let (n_llm_h, n_llm_w, best_height, best_width) = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        g.downsample_ratio,
        g.max_n_token,
    )?;
    let (n_vit_h, n_vit_w) = (best_height / p, best_width / p);
    if n_vit_h == 0 || n_vit_w == 0 {
        bail!("sizing {orig_width}x{orig_height} collapsed to an empty ViT grid");
    }

    // ★ The squash decision is taken from the ORIGINAL dimensions, not the clamped ones —
    // the reference tests `image.width >= ratio * image.height` on the decoded image, AFTER
    // `width` has already been overwritten by the clamp above.
    let squash = g.max_wh_ratio != 0 && orig_width >= g.max_wh_ratio * orig_height;

    Ok(DsImagePlan {
        best_width,
        best_height,
        n_vit_h,
        n_vit_w,
        n_llm_h,
        n_llm_w,
        squash,
    })
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
