// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash-Vision-Exp host f32 **ViT + aligner reference** (unit U1.3).
//!
//! Host reference, **not** a serving path: nothing here runs on a GPU and nothing here is
//! wired into a forward pass. Its job is to give the vision tower a checkable truth before
//! the CUDA path exists, so the Spark bring-up compares against DeepSeek's own arithmetic
//! rather than against a second Atlas implementation.
//!
//! # Where the truth comes from
//!
//! The truth source is DeepSeek's own `inference/vision.py` (`ViT` / `Aligner` /
//! `get_vision_cos_sin`) at revision `86f746b36186f0e567729a5c06a8c918caba82a9`. Every
//! numeric expectation in `vision_deepseek_ref/tests.rs` is an array that code produced,
//! recorded by `gen_vit_golden.py` into `fixtures/vit_tiny_golden.json`. No equation is
//! re-derived on the Python side; this module is the port, the golden is the judge.
//!
//! Weights and inputs are RNG-free (`layers::deepseek_v41_ref::fixed_value`, the in-tree
//! SSOT for the hash scheme), so the fixture carries only outputs. Regenerate it with
//!
//! ```text
//! /home/coder/.venvs/dsv4/bin/python3 \
//!     crates/spark-model/src/vision_deepseek_ref/gen_vit_golden.py
//! ```
//!
//! # Invariants worth stating once
//!
//! * The ViT's RMSNorm eps is its OWN `1e-6` ([`DsVitDims::eps`]) — never
//!   `ModelConfig::rms_norm_eps`, which carries `1e-20` unclamped for the text tower.
//!   Nothing in this module reads `ModelConfig`.
//! * Attention is FULL bidirectional over one image's patches, scaled by `1/√head_dim`;
//!   there is no causal mask and no window.
//! * Every derived dimension comes off [`DsVitDims`] ([`DsVitDims::head_dim`],
//!   [`DsVitDims::rope_dim`], [`DsVitDims::patch_in`]), never from a literal at a call site.
//! * Dot products accumulate in `f64` (see `mathf32`); shapes are asserted, never inferred.

pub mod load;
mod mathf32;

use mathf32::{gelu_erf, linear, rms_norm, silu, softmax_inplace};

pub use load::load_weights_from_safetensors;

/// Tiny/production dimensions of the vision tower, from the checkpoint's `vision_*` config.
#[derive(Debug, Clone)]
pub struct DsVitDims {
    pub layers: usize,
    pub dim: usize,
    pub heads: usize,
    pub inter: usize,
    pub patch: usize,
    pub ds: usize,
    pub out_dim: usize,
    pub rope_theta: f64,
    pub eps: f32,
}

impl DsVitDims {
    /// `dim / heads`.
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// `dim / heads / 2` — the 2D RoPE width (half height angles, half width angles).
    pub fn rope_dim(&self) -> usize {
        self.dim / self.heads / 2
    }

    /// `3 * patch * patch` — one patch's flattened pixel count.
    pub fn patch_in(&self) -> usize {
        3 * self.patch * self.patch
    }

    /// Fail loudly on dimensions the reference cannot express.
    fn check(&self) {
        assert!(self.layers > 0, "DsVitDims: layers must be positive");
        assert!(self.heads > 0 && self.dim > 0, "DsVitDims: dim/heads");
        assert!(
            self.dim.is_multiple_of(self.heads),
            "DsVitDims: dim % heads"
        );
        assert!(
            self.head_dim().is_multiple_of(2),
            "DsVitDims: head_dim must be even (rotate-half)"
        );
        assert!(self.ds > 0, "DsVitDims: downsample ratio must be positive");
        assert!(self.patch > 0, "DsVitDims: patch size must be positive");
        assert!(self.inter > 0 && self.out_dim > 0, "DsVitDims: widths");
    }
}

/// One encoder block's parameters, flattened row-major exactly as the checkpoint stores them.
#[derive(Debug, Clone)]
pub struct DsVitBlockWeights {
    pub norm1: Vec<f32>,
    /// `[3·dim, dim]` row-major.
    pub wqkv_w: Vec<f32>,
    pub wqkv_b: Vec<f32>,
    pub wo_w: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub norm2: Vec<f32>,
    /// `[2·inter, dim]` row-major (gate then up).
    pub w1: Vec<f32>,
    /// `[dim, inter]` row-major.
    pub w2: Vec<f32>,
}

/// The whole vision tower plus the aligner.
#[derive(Debug, Clone)]
pub struct DsVitWeights {
    /// `[dim, 3·p·p]` row-major.
    pub patch_w: Vec<f32>,
    pub patch_b: Vec<f32>,
    pub blocks: Vec<DsVitBlockWeights>,
    pub norm: Vec<f32>,
    /// `[out_dim, dim·ds²]` row-major.
    pub aligner_w1: Vec<f32>,
    pub aligner_b1: Vec<f32>,
    /// `[out_dim, out_dim]` row-major.
    pub aligner_w2: Vec<f32>,
    pub aligner_b2: Vec<f32>,
}

impl DsVitWeights {
    /// Every buffer length against `d` — a mis-shaped `DsVitWeights` is a loud failure here,
    /// never a silently truncated matrix deeper in.
    fn check_vit(&self, d: &DsVitDims) {
        let (dim, inter) = (d.dim, d.inter);
        assert_eq!(self.patch_w.len(), dim * d.patch_in(), "patch_embed.weight");
        assert_eq!(self.patch_b.len(), dim, "patch_embed.bias");
        assert_eq!(self.blocks.len(), d.layers, "block count");
        for (i, b) in self.blocks.iter().enumerate() {
            assert_eq!(b.norm1.len(), dim, "block {i} norm1");
            assert_eq!(b.wqkv_w.len(), 3 * dim * dim, "block {i} wqkv.weight");
            assert_eq!(b.wqkv_b.len(), 3 * dim, "block {i} wqkv.bias");
            assert_eq!(b.wo_w.len(), dim * dim, "block {i} wo.weight");
            assert_eq!(b.wo_b.len(), dim, "block {i} wo.bias");
            assert_eq!(b.norm2.len(), dim, "block {i} norm2");
            assert_eq!(b.w1.len(), 2 * inter * dim, "block {i} mlp.w1");
            assert_eq!(b.w2.len(), dim * inter, "block {i} mlp.w2");
        }
        assert_eq!(self.norm.len(), dim, "vision.norm");
    }

    /// The aligner half of the same check.
    fn check_aligner(&self, d: &DsVitDims) {
        let unshuffled = d.dim * d.ds * d.ds;
        assert_eq!(
            self.aligner_w1.len(),
            d.out_dim * unshuffled,
            "aligner.w1.weight"
        );
        assert_eq!(self.aligner_b1.len(), d.out_dim, "aligner.w1.bias");
        assert_eq!(
            self.aligner_w2.len(),
            d.out_dim * d.out_dim,
            "aligner.w2.weight"
        );
        assert_eq!(self.aligner_b2.len(), d.out_dim, "aligner.w2.bias");
    }
}

/// 2D RoPE tables, `[n_h·n_w, rope_dim]` each, row-major with `h` outer and `w` inner. The
/// first `rope_dim/2` columns carry `h·inv_freq`, the second half `w·inv_freq`, with
/// `inv_freq[j] = theta^(−2j/rope_dim)` — the reference's `get_vision_cos_sin`, minus its
/// trailing `unsqueeze(1)` (a head broadcast; every head reads the same row here).
pub fn rope_cos_sin(n_h: usize, n_w: usize, rope_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    assert!(n_h > 0 && n_w > 0, "rope_cos_sin: empty grid");
    assert!(
        rope_dim >= 2 && rope_dim.is_multiple_of(2),
        "rope_cos_sin: rope_dim must be even and positive"
    );
    let half = rope_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f64 / rope_dim as f64))
        .collect();
    let mut cos = Vec::with_capacity(n_h * n_w * rope_dim);
    let mut sin = Vec::with_capacity(n_h * n_w * rope_dim);
    for h in 0..n_h {
        for w in 0..n_w {
            for pos in [h as f64, w as f64] {
                for f in &inv_freq {
                    let angle = pos * f;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
    }
    (cos, sin)
}

/// Rotate-half in place over one head's `[2·rope_dim]` slice, against row `p` of the tables.
fn apply_rotary(head: &mut [f32], cos: &[f32], sin: &[f32], p: usize, rope_dim: usize) {
    let (c, s) = (
        &cos[p * rope_dim..(p + 1) * rope_dim],
        &sin[p * rope_dim..(p + 1) * rope_dim],
    );
    for j in 0..rope_dim {
        let (x1, x2) = (head[j] as f64, head[rope_dim + j] as f64);
        let (cj, sj) = (c[j] as f64, s[j] as f64);
        head[j] = (x1 * cj - x2 * sj) as f32;
        head[rope_dim + j] = (x2 * cj + x1 * sj) as f32;
    }
}

/// One block's attention: RMSNorm → fused QKV → rotary on q/k → full bidirectional softmax
/// attention at `1/√head_dim` → `wo`, accumulated into `x`.
fn block_attention(
    x: &mut [f32],
    b: &DsVitBlockWeights,
    d: &DsVitDims,
    rope: &(Vec<f32>, Vec<f32>),
    n: usize,
) {
    let (dim, heads, head_dim, rope_dim) = (d.dim, d.heads, d.head_dim(), d.rope_dim());
    let (mut q, mut k, mut v) = (
        vec![0f32; n * dim],
        vec![0f32; n * dim],
        vec![0f32; n * dim],
    );
    for p in 0..n {
        let h = rms_norm(&x[p * dim..(p + 1) * dim], &b.norm1, d.eps);
        let qkv = linear(&b.wqkv_w, Some(&b.wqkv_b), &h, dim);
        q[p * dim..(p + 1) * dim].copy_from_slice(&qkv[..dim]);
        k[p * dim..(p + 1) * dim].copy_from_slice(&qkv[dim..2 * dim]);
        v[p * dim..(p + 1) * dim].copy_from_slice(&qkv[2 * dim..]);
        for hd in 0..heads {
            let at = p * dim + hd * head_dim;
            apply_rotary(&mut q[at..at + head_dim], &rope.0, &rope.1, p, rope_dim);
            apply_rotary(&mut k[at..at + head_dim], &rope.0, &rope.1, p, rope_dim);
        }
    }
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut attn = vec![0f32; n * dim];
    let mut scores = vec![0f64; n];
    for hd in 0..heads {
        let off = hd * head_dim;
        for i in 0..n {
            let qi = &q[i * dim + off..i * dim + off + head_dim];
            for (j, score) in scores.iter_mut().enumerate() {
                let kj = &k[j * dim + off..j * dim + off + head_dim];
                *score = qi
                    .iter()
                    .zip(kj)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * scale;
            }
            softmax_inplace(&mut scores);
            let out = &mut attn[i * dim + off..i * dim + off + head_dim];
            for (j, w) in scores.iter().enumerate() {
                let vj = &v[j * dim + off..j * dim + off + head_dim];
                for (o, s) in out.iter_mut().zip(vj) {
                    *o = (*o as f64 + w * *s as f64) as f32;
                }
            }
        }
    }
    for p in 0..n {
        let proj = linear(&b.wo_w, Some(&b.wo_b), &attn[p * dim..(p + 1) * dim], dim);
        for (dst, add) in x[p * dim..(p + 1) * dim].iter_mut().zip(&proj) {
            *dst += *add;
        }
    }
}

/// One block's SwiGLU MLP: RMSNorm → `w1` (gate ‖ up) → `silu(gate)·up` → `w2`, into `x`.
fn block_mlp(x: &mut [f32], b: &DsVitBlockWeights, d: &DsVitDims, n: usize) {
    let (dim, inter) = (d.dim, d.inter);
    for p in 0..n {
        let h = rms_norm(&x[p * dim..(p + 1) * dim], &b.norm2, d.eps);
        let gu = linear(&b.w1, None, &h, dim);
        let act: Vec<f32> = (0..inter).map(|i| silu(gu[i]) * gu[inter + i]).collect();
        let down = linear(&b.w2, None, &act, inter);
        for (dst, add) in x[p * dim..(p + 1) * dim].iter_mut().zip(&down) {
            *dst += *add;
        }
    }
}

/// The encoder tower: `[n_h·n_w, dim]` from `[n_h·n_w, 3·p·p]` patches, whose inner order is
/// `(c, y, x)` — exactly what `PatchEmbed.forward`'s `x.flatten(1)` feeds its `proj`.
pub fn vit_forward(
    w: &DsVitWeights,
    d: &DsVitDims,
    patches: &[f32],
    n_h: usize,
    n_w: usize,
) -> Vec<f32> {
    d.check();
    w.check_vit(d);
    assert!(n_h > 0 && n_w > 0, "vit_forward: empty patch grid");
    let (n, dim, patch_in) = (n_h * n_w, d.dim, d.patch_in());
    assert_eq!(
        patches.len(),
        n * patch_in,
        "vit_forward: expected {n}·{patch_in} patch pixels for a {n_h}x{n_w} grid"
    );

    let mut x = vec![0f32; n * dim];
    for p in 0..n {
        let row = linear(
            &w.patch_w,
            Some(&w.patch_b),
            &patches[p * patch_in..(p + 1) * patch_in],
            patch_in,
        );
        x[p * dim..(p + 1) * dim].copy_from_slice(&row);
    }

    // Computed once outside the block loop, as `ViT.forward` does.
    let rope = rope_cos_sin(n_h, n_w, d.rope_dim(), d.rope_theta);
    for b in &w.blocks {
        block_attention(&mut x, b, d, &rope, n);
        block_mlp(&mut x, b, d, n);
    }

    let mut out = vec![0f32; n * dim];
    for p in 0..n {
        let row = rms_norm(&x[p * dim..(p + 1) * dim], &w.norm, d.eps);
        out[p * dim..(p + 1) * dim].copy_from_slice(&row);
    }
    out
}

/// The reference's `permute` + `F.pad` + `F.unfold`: `[n_llm_h·n_llm_w, dim·ds²]`, slot index
/// `c·ds² + kh·ds + kw`, zero-padded where the grid is short on either axis.
pub fn pixel_unshuffle(vit_out: &[f32], n_h: usize, n_w: usize, dim: usize, ds: usize) -> Vec<f32> {
    assert!(n_h > 0 && n_w > 0 && dim > 0 && ds > 0, "unshuffle: shape");
    assert_eq!(
        vit_out.len(),
        n_h * n_w * dim,
        "unshuffle: expected {n_h}·{n_w}·{dim} values"
    );
    let (llm_h, llm_w) = (n_h.div_ceil(ds), n_w.div_ceil(ds));
    let block = ds * ds;
    let mut out = vec![0f32; llm_h * llm_w * dim * block];
    for oh in 0..llm_h {
        for ow in 0..llm_w {
            let row = &mut out[(oh * llm_w + ow) * dim * block..][..dim * block];
            for kh in 0..ds {
                let sh = oh * ds + kh;
                if sh >= n_h {
                    continue;
                }
                for kw in 0..ds {
                    let sw = ow * ds + kw;
                    if sw >= n_w {
                        continue;
                    }
                    let src = &vit_out[(sh * n_w + sw) * dim..][..dim];
                    for (c, value) in src.iter().enumerate() {
                        row[c * block + kh * ds + kw] = *value;
                    }
                }
            }
        }
    }
    out
}

/// The aligner: unshuffle → `w1` → GELU (exact erf, **not** the tanh approximation) → `w2`;
/// `[n_llm_h·n_llm_w, out_dim]`.
pub fn aligner_forward(
    w: &DsVitWeights,
    d: &DsVitDims,
    vit_out: &[f32],
    n_h: usize,
    n_w: usize,
) -> Vec<f32> {
    d.check();
    w.check_aligner(d);
    let unshuffled = pixel_unshuffle(vit_out, n_h, n_w, d.dim, d.ds);
    let in_dim = d.dim * d.ds * d.ds;
    let rows = unshuffled.len() / in_dim;
    let mut out = vec![0f32; rows * d.out_dim];
    for r in 0..rows {
        let hidden = linear(
            &w.aligner_w1,
            Some(&w.aligner_b1),
            &unshuffled[r * in_dim..(r + 1) * in_dim],
            in_dim,
        );
        let activated: Vec<f32> = hidden.into_iter().map(gelu_erf).collect();
        let row = linear(&w.aligner_w2, Some(&w.aligner_b2), &activated, d.out_dim);
        out[r * d.out_dim..(r + 1) * d.out_dim].copy_from_slice(&row);
    }
    out
}

/// `aligner_forward ∘ vit_forward`, mirroring the reference's own composition.
pub fn encode_image(
    w: &DsVitWeights,
    d: &DsVitDims,
    patches: &[f32],
    n_h: usize,
    n_w: usize,
) -> Vec<f32> {
    aligner_forward(w, d, &vit_forward(w, d, patches, n_h, n_w), n_h, n_w)
}

#[cfg(test)]
mod loader_tests;
#[cfg(test)]
mod structure_tests;
#[cfg(test)]
mod tests;
