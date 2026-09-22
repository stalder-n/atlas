// SPDX-License-Identifier: AGPL-3.0-only

//! Pillow-parity resampling: `Image.resize(BICUBIC)` and `ImageOps.pad(color=(127,127,127))`.
//!
//! The reference preprocessor is Pillow, so "reference parity" means Pillow's arithmetic,
//! not a nice bicubic. This is a port of `src/libImaging/Resample.c`
//! (`precompute_coeffs`, `normalize_coeffs_8bpc`, `ResampleHorizontal_8bpc`,
//! `ResampleVertical_8bpc`) and of `PIL/ImageOps.py` (`contain`, `pad`), with Pillow's
//! own outputs recorded in `fixtures/resample.json`.
//!
//! Three details are easy to get subtly wrong and are each pinned by a golden case:
//! the u8 QUANTISED intermediate between the two passes, the support widened by
//! `max(scale, 1)` on downscale, and Python's BANKER'S rounding in `contain`/`pad`.

/// Pillow's `PRECISION_BITS`: `32 - 8 - 2`.
const PRECISION_BITS: u32 = 22;
/// Bicubic's `a` coefficient. Pillow uses -0.5 (Catmull-Rom), not the -0.75 of OpenCV.
const BICUBIC_A: f64 = -0.5;
/// Bicubic's filter support, before the downscale widening.
const BICUBIC_SUPPORT: f64 = 2.0;
/// `ImageOps.pad(..., color=(127, 127, 127))` — the reference's fill.
const PAD_FILL: [u8; 3] = [127, 127, 127];

/// Pillow's `bicubic_filter`.
fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * BICUBIC_A
    } else {
        0.0
    }
}

/// One output position's contribution window: source start, count, and fixed-point weights.
struct Coeffs {
    /// `bounds[xx * 2 + 0]` — first source index.
    start: Vec<usize>,
    /// `bounds[xx * 2 + 1]` — number of source samples.
    count: Vec<usize>,
    /// Row-major `[out_size, ksize]`, already normalised and scaled by `2^PRECISION_BITS`.
    k: Vec<i32>,
    ksize: usize,
}

/// Pillow's `precompute_coeffs` followed by `normalize_coeffs_8bpc`, over the full axis
/// (`in0 = 0`, `in1 = in_size`).
fn precompute_coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    // ★ On DOWNSCALE the filter is stretched so it averages the whole source footprint;
    // on upscale it is not. Dropping this makes a 10x reduction alias, which the
    // 200x150 -> 14x14 golden exists to catch.
    let filterscale = scale.max(1.0);
    let support = BICUBIC_SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut starts = Vec::with_capacity(out_size);
    let mut counts = Vec::with_capacity(out_size);
    let mut k = vec![0i32; out_size * ksize];
    let mut weights = vec![0.0f64; ksize];

    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // C's `(int)` truncates toward zero; so does `as i64`.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size);
        let count = xmax.saturating_sub(xmin);

        let mut ww = 0.0f64;
        for (x, w) in weights.iter_mut().enumerate().take(count) {
            *w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            ww += *w;
        }
        for (x, &w) in weights.iter().enumerate().take(count) {
            let norm = if ww != 0.0 { w / ww } else { w };
            // `normalize_coeffs_8bpc`: round half AWAY FROM ZERO into 22-bit fixed point.
            let scaled = norm * (1i64 << PRECISION_BITS) as f64;
            k[xx * ksize + x] = if norm < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            };
        }
        starts.push(xmin);
        counts.push(count);
    }

    Coeffs {
        start: starts,
        count: counts,
        k,
        ksize,
    }
}

/// Pillow's `clip8`: arithmetic shift down out of fixed point, then saturate to u8.
fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `ResampleHorizontal_8bpc` — RGB8 `[sh, sw]` to RGB8 `[sh, dw]`.
fn resample_horizontal(src: &[u8], sw: usize, sh: usize, dw: usize) -> Vec<u8> {
    let c = precompute_coeffs(sw, dw);
    let mut out = vec![0u8; dw * sh * 3];
    for yy in 0..sh {
        for xx in 0..dw {
            let (xmin, count) = (c.start[xx], c.count[xx]);
            let k = &c.k[xx * c.ksize..xx * c.ksize + c.ksize];
            // The accumulator is seeded with 2^(PRECISION_BITS-1) so the final shift
            // rounds to nearest rather than truncating.
            let mut acc = [1i64 << (PRECISION_BITS - 1); 3];
            for x in 0..count {
                let px = (yy * sw + xmin + x) * 3;
                for (ch, a) in acc.iter_mut().enumerate() {
                    *a += src[px + ch] as i64 * k[x] as i64;
                }
            }
            let dst = (yy * dw + xx) * 3;
            for ch in 0..3 {
                out[dst + ch] = clip8(acc[ch]);
            }
        }
    }
    out
}

/// `ResampleVertical_8bpc` — RGB8 `[sh, w]` to RGB8 `[dh, w]`.
fn resample_vertical(src: &[u8], w: usize, sh: usize, dh: usize) -> Vec<u8> {
    let c = precompute_coeffs(sh, dh);
    let mut out = vec![0u8; w * dh * 3];
    for yy in 0..dh {
        let (ymin, count) = (c.start[yy], c.count[yy]);
        let k = &c.k[yy * c.ksize..yy * c.ksize + c.ksize];
        for xx in 0..w {
            let mut acc = [1i64 << (PRECISION_BITS - 1); 3];
            for y in 0..count {
                let px = ((ymin + y) * w + xx) * 3;
                for (ch, a) in acc.iter_mut().enumerate() {
                    *a += src[px + ch] as i64 * k[y] as i64;
                }
            }
            let dst = (yy * w + xx) * 3;
            for ch in 0..3 {
                out[dst + ch] = clip8(acc[ch]);
            }
        }
    }
    out
}

/// PIL BICUBIC (`a = -0.5`, support widened by `max(scale, 1)` on downscale, separable
/// two-pass with a u8 intermediate). RGB8 in, RGB8 out.
pub fn resize_pil_bicubic_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), sw * sh * 3, "source is not RGB8 [sh, sw]");
    // `Image.resize`: `if self.size == size and box == (0, 0) + self.size: return self.copy()`
    // — an exact copy, no filter, no rounding.
    if (dw, dh) == (sw, sh) {
        return src.to_vec();
    }
    if dw == 0 || dh == 0 || sw == 0 || sh == 0 {
        return vec![0u8; dw * dh * 3];
    }
    // `ImagingResampleInner` runs the horizontal pass first and writes an 8-bit temporary;
    // the vertical pass then reads THOSE QUANTISED values. Keeping the intermediate in f32
    // would be more accurate and less like Pillow.
    let (mid, mid_w) = if dw != sw {
        (resample_horizontal(src, sw, sh, dw), dw)
    } else {
        (src.to_vec(), sw)
    };
    if dh != sh {
        resample_vertical(&mid, mid_w, sh, dh)
    } else {
        mid
    }
}

/// Python's `round()` — banker's rounding, which `ImageOps.contain`/`pad` rely on for the
/// contained dimension and the paste offset. Half-away-from-zero here is a silent
/// one-pixel shift of the whole image inside its gray frame.
pub fn py_round_half_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `ImageOps.contain`: the largest box with the source aspect ratio that fits `(dw, dh)`.
fn contain_size(sw: usize, sh: usize, dw: usize, dh: usize) -> (usize, usize) {
    let im_ratio = sw as f64 / sh as f64;
    let dest_ratio = dw as f64 / dh as f64;
    let mut size = (dw, dh);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let new_height = py_round_half_even(sh as f64 / sw as f64 * dw as f64);
            if new_height != dh as i64 {
                // `.max(1)`: PIL would accept a zero-height box here. Unreachable while the
                // caller's `squash` branch intercepts w >= 8h, but this is host code fed by
                // an HTTP body, so it degenerates to one row instead of an empty buffer.
                size = (dw, (new_height.max(1)) as usize);
            }
        } else {
            let new_width = py_round_half_even(sw as f64 / sh as f64 * dh as f64);
            if new_width != dw as i64 {
                size = ((new_width.max(1)) as usize, dh);
            }
        }
    }
    size
}

/// `ImageOps.pad(im, (dw, dh), color=(127,127,127))`: contain, centre, gray fill.
pub fn pad_contain_gray(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), sw * sh * 3, "source is not RGB8 [sh, sw]");
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return PAD_FILL.repeat(dw * dh);
    }
    let (rw, rh) = contain_size(sw, sh, dw, dh);
    let resized = resize_pil_bicubic_rgb(src, sw, sh, rw, rh);
    if (rw, rh) == (dw, dh) {
        return resized;
    }

    let mut out = PAD_FILL.repeat(dw * dh);
    // `pad` offsets along ONE axis only, chosen by `if resized.width != size[0]`, with
    // `centering = (0.5, 0.5)` and Python's `round`.
    let (ox, oy) = if rw != dw {
        (py_round_half_even((dw - rw) as f64 * 0.5) as usize, 0usize)
    } else {
        (0usize, py_round_half_even((dh - rh) as f64 * 0.5) as usize)
    };
    for y in 0..rh.min(dh.saturating_sub(oy)) {
        for x in 0..rw.min(dw.saturating_sub(ox)) {
            let s = (y * rw + x) * 3;
            let d = ((y + oy) * dw + x + ox) * 3;
            out[d..d + 3].copy_from_slice(&resized[s..s + 3]);
        }
    }
    out
}

#[cfg(test)]
#[path = "resample_tests.rs"]
mod tests;
