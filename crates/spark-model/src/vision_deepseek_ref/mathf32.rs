// SPDX-License-Identifier: AGPL-3.0-only

//! The f32 primitives the DeepSeek-V4 ViT/aligner reference is built from.
//!
//! Private to `vision_deepseek_ref`: these are host maths, not a serving path. Every dot
//! product accumulates in `f64` and is stored back as `f32` — torch's own f32 reduction
//! differs by ~1e-7, two orders of magnitude under the golden's 1e-4 bar, and the wider
//! accumulator removes summation order from the comparison entirely.

/// `y[o] = Σ_i w[o·in_dim + i]·x[i] (+ b[o])` — a row-major `[out_dim, in_dim]` matrix times
/// one vector, the layout `nn.Linear` stores its weight in. `out_dim` is `w.len() / in_dim`.
pub(super) fn linear(w: &[f32], b: Option<&[f32]>, x: &[f32], in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim, "linear: input width");
    assert!(
        in_dim > 0 && w.len().is_multiple_of(in_dim),
        "linear: weight shape"
    );
    let out_dim = w.len() / in_dim;
    if let Some(bias) = b {
        assert_eq!(bias.len(), out_dim, "linear: bias width");
    }
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let acc: f64 = row
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>()
                + b.map_or(0.0, |bias| bias[o] as f64);
            acc as f32
        })
        .collect()
}

/// `g ⊙ (x · rsqrt(mean(x²) + eps))` — the reference's `vision.py::RMSNorm`, whose `eps` is
/// its own `1e-6` and never the text tower's `ModelConfig::rms_norm_eps`.
pub(super) fn rms_norm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), g.len(), "rms_norm: width");
    assert!(!x.is_empty(), "rms_norm: empty row");
    let mean = x.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();
    x.iter()
        .zip(g)
        .map(|(v, w)| (*v as f64 * scale * *w as f64) as f32)
        .collect()
}

/// `x · sigmoid(x)`, evaluated in f64.
pub(super) fn silu(x: f32) -> f32 {
    let v = x as f64;
    (v / (1.0 + (-v).exp())) as f32
}

/// In-place softmax over a row, max-shifted before exponentiating.
pub(super) fn softmax_inplace(row: &mut [f64]) {
    assert!(!row.is_empty(), "softmax: empty row");
    let max = row.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v));
    let mut total = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    let inv = 1.0 / total;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Numerical Recipes' rational `erfc`, fractional error ≤ 1.2e-7 everywhere. Through
/// [`gelu_erf`] that is an absolute error below ~2.4e-7·|x|/2, i.e. ~1e-6 over the
/// activation range this reference sees — 100× under the golden's 1e-4 bar.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let ans = t * (-z * z + poly).exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// `erf(x) = 1 − erfc(x)`.
fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Exact-erf GELU — `0.5·x·(1 + erf(x/√2))`, torch's `F.gelu` default (`approximate="none"`),
/// which is what `Aligner.forward` calls. NOT the tanh approximation.
pub(super) fn gelu_erf(x: f32) -> f32 {
    let v = x as f64;
    (0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2))) as f32
}
