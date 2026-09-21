// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance tests for the DeepSeek-V4 ViT/aligner **safetensors loader** (unit U1.3).
//!
//! The loader is the one file-I/O site in this unit, so it is pinned here against a
//! host-synthesised checkpoint written in the real container format (8-byte LE header
//! length, JSON header with `data_offsets`, BF16 payload) under the real tensor names. The
//! real 48-shard checkpoint is exercised by the `#[ignore]`d smoke test at the bottom, which
//! only the Spark runs.

use super::tests::{ck, dims, fixture};
use super::*;
use crate::layers::deepseek_v41_ref::fixed_value;
use serde_json::Value;
use std::path::{Path, PathBuf};
// --------------------------------------------------------------------------------------
// the safetensors loader (host-synthesised checkpoint; the real one is the ignored smoke)
// --------------------------------------------------------------------------------------

/// The 23 `vision.*` / `aligner.*` tensors the loader must find, with the shape each must
/// have — the same names the real checkpoint's index carries.
fn ckpt_tensors(d: &DsVitDims) -> Vec<(String, Vec<usize>)> {
    let mut t = vec![
        (
            "vision.patch_embed.proj.weight".into(),
            vec![d.dim, d.patch_in()],
        ),
        ("vision.patch_embed.proj.bias".into(), vec![d.dim]),
    ];
    for i in 0..d.layers {
        t.push((format!("vision.blocks.{i}.norm1.weight"), vec![d.dim]));
        t.push((
            format!("vision.blocks.{i}.attn.wqkv.weight"),
            vec![3 * d.dim, d.dim],
        ));
        t.push((format!("vision.blocks.{i}.attn.wqkv.bias"), vec![3 * d.dim]));
        t.push((
            format!("vision.blocks.{i}.attn.wo.weight"),
            vec![d.dim, d.dim],
        ));
        t.push((format!("vision.blocks.{i}.attn.wo.bias"), vec![d.dim]));
        t.push((format!("vision.blocks.{i}.norm2.weight"), vec![d.dim]));
        t.push((
            format!("vision.blocks.{i}.mlp.w1.weight"),
            vec![2 * d.inter, d.dim],
        ));
        t.push((
            format!("vision.blocks.{i}.mlp.w2.weight"),
            vec![d.dim, d.inter],
        ));
    }
    t.push(("vision.norm.weight".into(), vec![d.dim]));
    t.push((
        "aligner.w1.weight".into(),
        vec![d.out_dim, d.dim * d.ds * d.ds],
    ));
    t.push(("aligner.w1.bias".into(), vec![d.out_dim]));
    t.push(("aligner.w2.weight".into(), vec![d.out_dim, d.out_dim]));
    t.push(("aligner.w2.bias".into(), vec![d.out_dim]));
    t
}

/// Deterministic BF16 content for a synthetic tensor.
fn synth(name: &str, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| half::bf16::from_f32(fixed_value(name, i, 0.5, 0.0)).to_f32())
        .collect()
}

/// Write one safetensors shard: 8-byte LE header length, JSON header, then the data block.
fn write_shard(path: &Path, tensors: &[(String, Vec<usize>, String)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, dtype) in tensors {
        let n: usize = shape.iter().product();
        let start = blob.len();
        for v in synth(name, n) {
            blob.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, blob.len()]}),
        );
    }
    let text = serde_json::to_vec(&Value::Object(header)).expect("header serialises");
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&text);
    out.extend_from_slice(&blob);
    std::fs::write(path, out).expect("shard written");
}

/// A scratch directory holding a synthetic checkpoint; `index` selects the index-json path
/// (what the real checkpoint uses) or the bare `model.safetensors` fallback.
fn synth_ckpt(
    tag: &str,
    d: &DsVitDims,
    index: bool,
    tweak: impl Fn(&mut Vec<(String, Vec<usize>, String)>),
) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "dsv4-vit-{tag}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut tensors: Vec<(String, Vec<usize>, String)> = ckpt_tensors(d)
        .into_iter()
        .map(|(n, s)| (n, s, "BF16".to_string()))
        .collect();
    tweak(&mut tensors);
    let shard = if index {
        "model-00001-of-00001.safetensors"
    } else {
        "model.safetensors"
    };
    write_shard(&dir.join(shard), &tensors);
    if index {
        let map: serde_json::Map<String, Value> = tensors
            .iter()
            .map(|(n, _, _)| (n.clone(), Value::String(shard.to_string())))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).expect("index"),
        )
        .expect("index written");
    }
    dir
}

/// The index-driven happy path: every buffer has the length `DsVitDims` implies and the BF16
/// bytes land as the exact f32 they widen to.
#[test]
fn load_weights_reads_the_indexed_vision_and_aligner_tensors() {
    let d = dims(&fixture());
    let dir = synth_ckpt("ok", &d, true, |_| {});
    let w = load_weights_from_safetensors(&dir, &d).expect("synthetic checkpoint loads");
    assert_eq!(w.patch_w.len(), d.dim * d.patch_in());
    assert_eq!(w.patch_b.len(), d.dim);
    assert_eq!(w.blocks.len(), d.layers);
    for (i, b) in w.blocks.iter().enumerate() {
        assert_eq!(b.norm1.len(), d.dim, "block {i} norm1");
        assert_eq!(b.wqkv_w.len(), 3 * d.dim * d.dim, "block {i} wqkv");
        assert_eq!(b.wqkv_b.len(), 3 * d.dim);
        assert_eq!(b.wo_w.len(), d.dim * d.dim);
        assert_eq!(b.wo_b.len(), d.dim);
        assert_eq!(b.norm2.len(), d.dim);
        assert_eq!(b.w1.len(), 2 * d.inter * d.dim, "block {i} mlp.w1");
        assert_eq!(b.w2.len(), d.dim * d.inter, "block {i} mlp.w2");
    }
    assert_eq!(w.norm.len(), d.dim);
    assert_eq!(w.aligner_w1.len(), d.out_dim * d.dim * d.ds * d.ds);
    assert_eq!(w.aligner_b1.len(), d.out_dim);
    assert_eq!(w.aligner_w2.len(), d.out_dim * d.out_dim);
    assert_eq!(w.aligner_b2.len(), d.out_dim);
    assert_eq!(
        w.aligner_w2,
        synth("aligner.w2.weight", d.out_dim * d.out_dim),
        "BF16 → f32 widening must be exact"
    );
    assert_eq!(
        w.blocks[1].wqkv_b,
        synth("vision.blocks.1.attn.wqkv.bias", 3 * d.dim)
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// No index file: the single-shard layout is the documented fallback.
#[test]
fn load_weights_falls_back_to_a_single_shard() {
    let d = dims(&fixture());
    let dir = synth_ckpt("single", &d, false, |_| {});
    let w = load_weights_from_safetensors(&dir, &d).expect("single-shard checkpoint loads");
    assert_eq!(w.patch_b, synth("vision.patch_embed.proj.bias", d.dim));
    std::fs::remove_dir_all(&dir).ok();
}

/// PCND: nothing is inferred and nothing is defaulted — a wrong shape, a wrong dtype, a
/// missing tensor and a missing directory each fail fast, naming what was wrong.
#[test]
fn load_weights_fails_fast_and_names_the_offender() {
    let d = dims(&fixture());

    let err = load_weights_from_safetensors(Path::new("/nonexistent/dsv4-ckpt"), &d)
        .expect_err("a missing directory must be an error");
    assert!(
        format!("{err:#}").contains("/nonexistent/dsv4-ckpt"),
        "the error must name the directory: {err:#}"
    );

    let dir = synth_ckpt("badshape", &d, true, |t| {
        let e = t
            .iter_mut()
            .find(|(n, _, _)| n == "vision.norm.weight")
            .expect("present");
        e.1 = vec![d.dim + 1];
    });
    let err = load_weights_from_safetensors(&dir, &d).expect_err("a wrong shape must be an error");
    assert!(
        format!("{err:#}").contains("vision.norm.weight"),
        "the error must name the tensor: {err:#}"
    );
    std::fs::remove_dir_all(&dir).ok();

    let dir = synth_ckpt("baddtype", &d, true, |t| {
        let e = t
            .iter_mut()
            .find(|(n, _, _)| n == "aligner.w1.bias")
            .expect("present");
        e.2 = "F16".to_string();
    });
    let err = load_weights_from_safetensors(&dir, &d).expect_err("F16 must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("aligner.w1.bias") && msg.contains("F16"),
        "name tensor + dtype: {msg}"
    );
    std::fs::remove_dir_all(&dir).ok();

    let dir = synth_ckpt("missing", &d, true, |t| {
        t.retain(|(n, _, _)| n != "vision.blocks.0.mlp.w2.weight");
    });
    let err = load_weights_from_safetensors(&dir, &d).expect_err("a missing tensor is an error");
    assert!(
        format!("{err:#}").contains("vision.blocks.0.mlp.w2.weight"),
        "the error must name the absent tensor: {err:#}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// --------------------------------------------------------------------------------------
// Spark-only: the real checkpoint
// --------------------------------------------------------------------------------------

/// Runs on the Spark, against the real checkpoint: `DSV4_VISION_CKPT=<dir> cargo test -p
/// spark-model --lib vision_deepseek_ref -- --ignored --nocapture`. Prints the fp64 checksum
/// the bring-up handoff quotes.
#[test]
#[ignore = "needs the real checkpoint in $DSV4_VISION_CKPT (Spark only)"]
fn reference_real_checkpoint_smoke() {
    let Ok(dir) = std::env::var("DSV4_VISION_CKPT") else {
        eprintln!("skipped: set DSV4_VISION_CKPT to the checkpoint directory");
        return;
    };
    // The checkpoint's own config.json: vision_n_layers, vision_dim, vision_n_heads,
    // vision_inter_dim, vision_patch_size, vision_downsample_ratio, dim, vision_rope_theta.
    let d = DsVitDims {
        layers: 32,
        dim: 1024,
        heads: 16,
        inter: 2816,
        patch: 14,
        ds: 3,
        out_dim: 4096,
        rope_theta: 10000.0,
        eps: 1e-6,
    };
    let w = load_weights_from_safetensors(Path::new(&dir), &d).expect("vision.* tensors load");
    assert_eq!(w.blocks.len(), d.layers);
    assert_eq!(w.patch_w.len(), d.dim * d.patch_in());
    assert_eq!(w.aligner_w1.len(), d.out_dim * d.dim * d.ds * d.ds);
    let (n_h, n_w) = (6usize, 6usize);
    let p: Vec<f32> = (0..(n_h * n_w * d.patch_in()) as u64)
        .map(|i| fixed_value("smoke.patches", i, 1.0, 0.0))
        .collect();
    let out = encode_image(&w, &d, &p, n_h, n_w);
    assert_eq!(out.len(), 4 * d.out_dim, "2x2 llm grid at ds=3");
    assert!(
        out.iter().all(|v| v.is_finite() && v.abs() < 1e3),
        "aligner output is sane"
    );
    println!(
        "encode_image checksum = {:.6}",
        ck(out.iter().map(|&v| v as f64))
    );
}
