// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the DeepSeek-V4 host f32 ViT/aligner reference (unit U1.3).
//!
//! Every numeric expectation is an array DeepSeek's OWN `inference/vision.py` produced at
//! revision `86f746b3…`, recorded by `gen_vit_golden.py` into `fixtures/vit_tiny_golden.json`.
//! Nothing here is hand-derived, so a passing test proves reference parity rather than
//! self-consistency. Weights and inputs are RNG-free: the fixture carries only the rule
//! (`weights_meta`), and the tests regenerate the tensors with the ds41 hash scheme
//! (`layers::deepseek_v41_ref`, the SSOT for `splitmix64`/`fnv1a64`/`fixed_value`).
//!
//! The comparison bar (`tolerance.abs` / `tolerance.rel`) is read from the fixture, so there
//! is exactly one place where the bar is stated.

use super::mathf32::{gelu_erf, linear};
use super::*;
use crate::layers::deepseek_v41_ref::{fixed_raw, fixed_value};
use crate::vision_deepseek::geometry;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/vit_tiny_golden.json");
const REVISION: &str = "86f746b36186f0e567729a5c06a8c918caba82a9";
const REPO: &str = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp";
const CAPTURES: [&str; 7] = [
    "rope_cos",
    "rope_sin",
    "patch_embed",
    "vit_out",
    "unshuffle",
    "aligner_hidden",
    "aligner_out",
];

// --------------------------------------------------------------------------------------
// fixture access
// --------------------------------------------------------------------------------------

pub(super) fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/vit_tiny_golden.json parses")
}

fn u(v: &Value, k: &str) -> usize {
    v.get(k)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("golden node is missing the integer field `{k}`")) as usize
}

fn fl(v: &Value, k: &str) -> f64 {
    v.get(k)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("golden node is missing the numeric field `{k}`"))
}

fn arr(v: &Value, k: &str) -> Vec<f64> {
    v.get(k)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("golden node is missing the array `{k}`"))
        .iter()
        .map(|x| x.as_f64().expect("numeric golden element"))
        .collect()
}

fn cases(f: &Value) -> &Vec<Value> {
    f["cases"].as_array().expect("fixture has a `cases` array")
}

pub(super) fn dims(f: &Value) -> DsVitDims {
    let d = &f["dims"];
    DsVitDims {
        layers: u(d, "layers"),
        dim: u(d, "dim"),
        heads: u(d, "heads"),
        inter: u(d, "inter"),
        patch: u(d, "patch"),
        ds: u(d, "ds"),
        out_dim: u(d, "out_dim"),
        rope_theta: fl(d, "rope_theta"),
        eps: fl(d, "eps") as f32,
    }
}

/// Regenerate one parameter from its committed rule — never from a hard-coded scale.
fn regen(f: &Value, name: &str) -> Vec<f32> {
    let m = &f["weights_meta"][name];
    assert!(!m.is_null(), "weights_meta has no `{name}`");
    let (n, scale, offset) = (u(m, "n"), fl(m, "scale"), fl(m, "offset"));
    (0..n as u64)
        .map(|i| fixed_value(name, i, scale, offset))
        .collect()
}

pub(super) fn weights(f: &Value) -> DsVitWeights {
    let d = dims(f);
    let blocks = (0..d.layers)
        .map(|i| DsVitBlockWeights {
            norm1: regen(f, &format!("vision.blocks.{i}.norm1.weight")),
            wqkv_w: regen(f, &format!("vision.blocks.{i}.attn.wqkv.weight")),
            wqkv_b: regen(f, &format!("vision.blocks.{i}.attn.wqkv.bias")),
            wo_w: regen(f, &format!("vision.blocks.{i}.attn.wo.weight")),
            wo_b: regen(f, &format!("vision.blocks.{i}.attn.wo.bias")),
            norm2: regen(f, &format!("vision.blocks.{i}.norm2.weight")),
            w1: regen(f, &format!("vision.blocks.{i}.mlp.w1.weight")),
            w2: regen(f, &format!("vision.blocks.{i}.mlp.w2.weight")),
        })
        .collect();
    DsVitWeights {
        patch_w: regen(f, "vision.patch_embed.proj.weight"),
        patch_b: regen(f, "vision.patch_embed.proj.bias"),
        blocks,
        norm: regen(f, "vision.norm.weight"),
        aligner_w1: regen(f, "aligner.w1.weight"),
        aligner_b1: regen(f, "aligner.w1.bias"),
        aligner_w2: regen(f, "aligner.w2.weight"),
        aligner_b2: regen(f, "aligner.w2.bias"),
    }
}

/// The case's input patches, `[n_h·n_w, 3·p·p]` row-major — the generator's own filler.
fn patches(c: &Value) -> Vec<f32> {
    let name = c["patches_name"].as_str().expect("patches_name");
    (0..u(c, "patches_n") as u64)
        .map(|i| fixed_value(name, i, 1.0, 0.0))
        .collect()
}

/// `sum(v[i] * (i + 1))` in f64 — the generator's `ck`.
pub(super) fn ck<I: IntoIterator<Item = f64>>(v: I) -> f64 {
    v.into_iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 + 1.0))
        .sum()
}

/// Absolute AND relative-to-peak comparison, with both bars read from the fixture.
#[track_caller]
fn check(what: &str, got: &[f32], want: &[f64], f: &Value) {
    let (abs, rel) = (fl(&f["tolerance"], "abs"), fl(&f["tolerance"], "rel"));
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0f64, |a, v| a.max(v.abs()));
    let bar = abs.min(rel * peak.max(f64::MIN_POSITIVE));
    let mut worst = 0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (*g as f64 - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= bar,
        "{what}: max abs diff {worst:e} > bar {bar:e} at index {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

// --------------------------------------------------------------------------------------
// the golden itself
// --------------------------------------------------------------------------------------

/// Pins the fixture: right checkpoint, right revision, the spec's tiny dims, all four grids
/// and every capture. A golden swapped for a weaker one fails here instead of silently
/// making every comparison below vacuous. (The one test that calls no production code.)
#[test]
fn golden_provenance_is_pinned() {
    let f = fixture();
    assert_eq!(f["reference"]["revision"], REVISION);
    assert_eq!(f["reference"]["repo"], REPO);
    assert!(
        f["reference"]["files"]["vision.py"]
            .as_str()
            .is_some_and(|s| s.len() == 64),
        "the fixture must record a sha256 of the reference vision.py"
    );
    assert!(
        f["generator"]
            .as_str()
            .is_some_and(|s| s.contains("gen_vit_golden.py")),
        "the fixture must record the generator invocation"
    );
    let d = dims(&f);
    assert_eq!(
        (d.layers, d.dim, d.heads, d.inter, d.patch, d.ds, d.out_dim),
        (2, 32, 2, 40, 2, 3, 16),
        "the spec's tiny dims"
    );
    assert_eq!(d.rope_theta, 10000.0);
    assert_eq!(
        d.eps, 1e-6,
        "the ViT's own eps, NEVER ModelConfig::rms_norm_eps"
    );
    assert_eq!(fl(&f["tolerance"], "abs"), 1e-4);
    assert_eq!(fl(&f["tolerance"], "rel"), 1e-3);

    let grids: Vec<(usize, usize)> = cases(&f)
        .iter()
        .map(|c| (u(c, "n_h"), u(c, "n_w")))
        .collect();
    assert_eq!(
        grids,
        vec![(5, 7), (3, 3), (1, 4), (6, 6)],
        "pad on both axes, on neither (twice), and a single-row grid"
    );
    for c in cases(&f) {
        for name in CAPTURES {
            let a = arr(c, name);
            assert!(
                !a.is_empty(),
                "case {:?} capture `{name}` is empty",
                c["n_h"]
            );
            assert!(
                a.iter().all(|v| v.is_finite()),
                "capture `{name}` has a non-finite value"
            );
        }
    }
    assert_eq!(
        f["weights_meta"].as_object().expect("weights_meta").len(),
        23,
        "patch_embed(2) + 2 blocks x 8 + final norm(1) + aligner(4)"
    );
}

/// The hash stream itself, before any activation is compared: a scheme drift is then
/// diagnosed as a scheme drift rather than as a broken ViT.
#[test]
fn rng_scheme_matches_the_generator() {
    let f = fixture();
    let p = &f["rng_probe"];
    let name = p["name"].as_str().expect("rng_probe.name");
    let raw: Vec<u64> = p["raw"]
        .as_array()
        .expect("rng_probe.raw")
        .iter()
        .map(|v| v.as_u64().expect("u64"))
        .collect();
    assert_eq!(raw.len(), 8);
    let got: Vec<u64> = (0..8).map(|i| fixed_raw(name, i)).collect();
    assert_eq!(
        got, raw,
        "splitmix64/fnv1a64 stream diverges from the generator"
    );
    let vals = arr(p, "values");
    let (scale, offset) = (fl(p, "scale"), fl(p, "offset"));
    let got: Vec<f64> = (0..8)
        .map(|i| fixed_value(name, i, scale, offset) as f64)
        .collect();
    assert_eq!(got, vals, "fixed_value diverges from the generator");
}

/// Every parameter and every input regenerates to the committed checksum, so the reference
/// below is fed exactly the tensors the reference ran on.
#[test]
fn weights_and_inputs_regenerate_to_the_golden_checksums() {
    let f = fixture();
    let mut checked = 0usize;
    for (name, m) in f["weights_meta"].as_object().expect("weights_meta") {
        let v = regen(&f, name);
        assert_eq!(v.len(), u(m, "n"), "{name}: element count");
        let want = fl(m, "ck");
        let got = ck(v.iter().map(|&x| x as f64));
        let tol = 1e-9 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{name}: regenerated ck {got} vs committed {want}"
        );
        checked += 1;
    }
    assert_eq!(checked, 23, "every ViT + aligner parameter is covered");
    for c in cases(&f) {
        let p = patches(c);
        assert_eq!(p.len(), u(c, "patches_n"));
        let want = fl(c, "patches_ck");
        let got = ck(p.iter().map(|&x| x as f64));
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "{}: regenerated input ck {got} vs committed {want}",
            c["patches_name"]
        );
    }
}

// --------------------------------------------------------------------------------------
// the port, against the reference's own arrays
// --------------------------------------------------------------------------------------

/// 2D RoPE: `h·inv_freq` in the first half of every row, `w·inv_freq` in the second, rows in
/// `h`-outer/`w`-inner order, `inv_freq[j] = theta^(-2j/rope_dim)`.
#[test]
fn rope_cos_sin_matches_the_reference() {
    let f = fixture();
    let d = dims(&f);
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let (cos, sin) = rope_cos_sin(n_h, n_w, d.rope_dim(), d.rope_theta);
        assert_eq!(
            cos.len(),
            n_h * n_w * d.rope_dim(),
            "cos shape for {n_h}x{n_w}"
        );
        check(
            &format!("rope_cos {n_h}x{n_w}"),
            &cos,
            &arr(c, "rope_cos"),
            &f,
        );
        check(
            &format!("rope_sin {n_h}x{n_w}"),
            &sin,
            &arr(c, "rope_sin"),
            &f,
        );
    }
}

/// The tower's first step on its own: `proj(x.flatten(1))` over `[n, 3·p·p]` patches whose
/// inner order is `(c, y, x)`. Localises a patch-embedding regression, which would otherwise
/// only surface as an opaque `vit_out` mismatch.
#[test]
fn patch_embed_matches_the_reference() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    let patch_in = d.patch_in();
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let p = patches(c);
        let mut got = Vec::with_capacity(n_h * n_w * d.dim);
        for row in p.chunks_exact(patch_in) {
            got.extend(linear(&w.patch_w, Some(&w.patch_b), row, patch_in));
        }
        check(
            &format!("patch_embed {n_h}x{n_w}"),
            &got,
            &arr(c, "patch_embed"),
            &f,
        );
    }
}

/// The whole encoder tower: patch embedding, per-block RMSNorm(eps 1e-6) → rotary QKV →
/// full bidirectional attention scaled by `1/√head_dim` → SwiGLU MLP, then the final norm.
#[test]
fn vit_forward_matches_the_reference() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let out = vit_forward(&w, &d, &patches(c), n_h, n_w);
        assert_eq!(
            out.len(),
            n_h * n_w * d.dim,
            "vit_forward shape for {n_h}x{n_w}"
        );
        check(
            &format!("vit_out {n_h}x{n_w}"),
            &out,
            &arr(c, "vit_out"),
            &f,
        );
    }
}

/// The reference's `permute` + `F.pad` + `F.unfold`: index `c·ds² + kh·ds + kw`, zero padding
/// on a short row or column. Checked against the tensor the reference's own `aligner.w1` saw.
#[test]
fn pixel_unshuffle_matches_the_reference() {
    let f = fixture();
    let d = dims(&f);
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let vit_out: Vec<f32> = arr(c, "vit_out").iter().map(|&v| v as f32).collect();
        let got = pixel_unshuffle(&vit_out, n_h, n_w, d.dim, d.ds);
        assert_eq!(
            got.len(),
            u(c, "n_llm_h") * u(c, "n_llm_w") * d.dim * d.ds * d.ds,
            "unshuffle shape for {n_h}x{n_w}"
        );
        check(
            &format!("unshuffle {n_h}x{n_w}"),
            &got,
            &arr(c, "unshuffle"),
            &f,
        );
    }
}

/// The aligner's first half on its own — `w1` then exact-erf GELU, against the tensor the
/// reference's own `aligner.w2` saw. Pins the GELU flavour (erf, not tanh) independently of
/// the second projection, which would otherwise mask or mimic the error.
#[test]
fn aligner_hidden_matches_the_reference() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    let in_dim = d.dim * d.ds * d.ds;
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let vit_out: Vec<f32> = arr(c, "vit_out").iter().map(|&v| v as f32).collect();
        let unshuffled = pixel_unshuffle(&vit_out, n_h, n_w, d.dim, d.ds);
        let mut got = Vec::with_capacity(unshuffled.len() / in_dim * d.out_dim);
        for row in unshuffled.chunks_exact(in_dim) {
            got.extend(
                linear(&w.aligner_w1, Some(&w.aligner_b1), row, in_dim)
                    .into_iter()
                    .map(gelu_erf),
            );
        }
        check(
            &format!("aligner_hidden {n_h}x{n_w}"),
            &got,
            &arr(c, "aligner_hidden"),
            &f,
        );
    }
}

/// The aligner: unshuffle → `w1` → exact-erf GELU → `w2`, with the row count agreeing with
/// the production geometry helper (`n_llm`), not with a second copy of the arithmetic.
#[test]
fn aligner_forward_matches_the_reference() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let vit_out: Vec<f32> = arr(c, "vit_out").iter().map(|&v| v as f32).collect();
        let out = aligner_forward(&w, &d, &vit_out, n_h, n_w);
        let rows = geometry::n_llm(n_h, d.ds) * geometry::n_llm(n_w, d.ds);
        assert_eq!(rows, u(c, "n_llm_h") * u(c, "n_llm_w"), "row count");
        assert_eq!(out.len(), rows * d.out_dim, "aligner shape for {n_h}x{n_w}");
        check(
            &format!("aligner_out {n_h}x{n_w}"),
            &out,
            &arr(c, "aligner_out"),
            &f,
        );
    }
}

/// `encode_image` is exactly the composition — bit equality, not a tolerance.
#[test]
fn encode_image_is_vit_then_aligner() {
    let f = fixture();
    let (d, w) = (dims(&f), weights(&f));
    for c in cases(&f) {
        let (n_h, n_w) = (u(c, "n_h"), u(c, "n_w"));
        let p = patches(c);
        let composed = aligner_forward(&w, &d, &vit_forward(&w, &d, &p, n_h, n_w), n_h, n_w);
        let direct = encode_image(&w, &d, &p, n_h, n_w);
        assert_eq!(
            direct, composed,
            "encode_image must be aligner∘vit for {n_h}x{n_w}"
        );
        check(
            &format!("encode_image {n_h}x{n_w}"),
            &direct,
            &arr(c, "aligner_out"),
            &f,
        );
    }
}
