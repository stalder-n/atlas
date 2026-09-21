// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors reader for the checkpoint's `vision.*` / `aligner.*` tensors.
//!
//! The container is parsed directly — 8-byte little-endian header length, JSON header,
//! `data_offsets` into the payload — rather than through the `safetensors` crate, which is
//! an *optional* `spark-model` dependency reachable only through the `metal-example` feature
//! and would additionally want the whole shard resident. A ranged `seek` + `read_exact`
//! touches only the vision tensors, which matters when the real checkpoint is 48 shards.
//!
//! SBIO note: this is the one file-I/O site in the unit. It is a loader — the category the
//! invariant exempts — reached only from the Spark validation and the `#[ignore]`d smoke
//! test; the compute functions in the parent module never touch the filesystem.
//!
//! PCND: nothing is inferred from the file and nothing is defaulted. Every tensor's dtype
//! and shape is checked against the caller's [`DsVitDims`], and any mismatch, absence or
//! truncation is an error naming the offending tensor.

use super::{DsVitBlockWeights, DsVitDims, DsVitWeights};
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Refuse a header claiming more than this, matching the runtime loader's own guard.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Every tensor the reference needs, with the shape `d` says it must have. This list is the
/// single place the checkpoint's naming is written down.
fn required_tensors(d: &DsVitDims) -> Vec<(String, Vec<usize>)> {
    let mut t = vec![
        (
            "vision.patch_embed.proj.weight".to_string(),
            vec![d.dim, d.patch_in()],
        ),
        ("vision.patch_embed.proj.bias".to_string(), vec![d.dim]),
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
    t.push(("vision.norm.weight".to_string(), vec![d.dim]));
    t.push((
        "aligner.w1.weight".to_string(),
        vec![d.out_dim, d.dim * d.ds * d.ds],
    ));
    t.push(("aligner.w1.bias".to_string(), vec![d.out_dim]));
    t.push(("aligner.w2.weight".to_string(), vec![d.out_dim, d.out_dim]));
    t.push(("aligner.w2.bias".to_string(), vec![d.out_dim]));
    t
}

/// Group the required tensors by the shard holding them — from the HF index, or the
/// single-file fallback — so each shard is opened once, in a deterministic order.
fn group_by_shard(
    dir: &Path,
    required: Vec<(String, Vec<usize>)>,
) -> Result<BTreeMap<PathBuf, Vec<(String, Vec<usize>)>>> {
    let index = dir.join("model.safetensors.index.json");
    let single = dir.join("model.safetensors");
    let weight_map = if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .with_context(|| format!("reading {}", index.display()))?;
        let mut doc: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", index.display()))?;
        let map = doc
            .get_mut("weight_map")
            .and_then(serde_json::Value::as_object_mut)
            .with_context(|| format!("{} has no `weight_map` object", index.display()))?;
        Some(std::mem::take(map))
    } else if single.is_file() {
        None
    } else {
        bail!(
            "no safetensors checkpoint in {}: neither model.safetensors.index.json nor \
             model.safetensors is present",
            dir.display()
        )
    };

    let mut by_shard: BTreeMap<PathBuf, Vec<(String, Vec<usize>)>> = BTreeMap::new();
    for (name, shape) in required {
        let shard = match &weight_map {
            Some(map) => {
                let Some(file) = map.get(&name).and_then(serde_json::Value::as_str) else {
                    bail!("{name}: absent from {}", index.display());
                };
                dir.join(file)
            }
            None => single.clone(),
        };
        by_shard.entry(shard).or_default().push((name, shape));
    }
    Ok(by_shard)
}

/// Byte width of a dtype this reader accepts.
fn dtype_width(name: &str, dtype: &str) -> Result<usize> {
    match dtype {
        "BF16" => Ok(2),
        "F32" => Ok(4),
        other => bail!("{name}: unsupported dtype {other} (expected BF16 or F32)"),
    }
}

/// Widen a raw tensor payload to f32.
fn widen(name: &str, dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => bail!("{name}: unsupported dtype {other} (expected BF16 or F32)"),
    }
}

/// Read one shard's JSON header and return it with the payload's start offset.
fn read_header(path: &Path, file: &mut File) -> Result<(serde_json::Value, u64)> {
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .with_context(|| format!("{}: truncated safetensors header length", path.display()))?;
    let header_len = u64::from_le_bytes(len);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!(
            "{}: implausible safetensors header length {header_len}",
            path.display()
        );
    }
    let mut raw = vec![0u8; header_len as usize];
    file.read_exact(&mut raw)
        .with_context(|| format!("{}: truncated safetensors header", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&raw)
        .with_context(|| format!("{}: safetensors header is not JSON", path.display()))?;
    Ok((header, 8 + header_len))
}

/// Pull the wanted tensors out of one shard, validating dtype and shape against `wanted`.
fn read_shard(
    path: &Path,
    wanted: &[(String, Vec<usize>)],
    into: &mut HashMap<String, Vec<f32>>,
) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("opening shard {}", path.display()))?;
    let (header, base) = read_header(path, &mut file)?;
    for (name, shape) in wanted {
        let Some(entry) = header.get(name) else {
            bail!("{name}: not in the header of {}", path.display());
        };
        let dtype = entry
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("{name}: header entry has no dtype"))?;
        let width = dtype_width(name, dtype)?;
        let got: Vec<usize> = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("{name}: header entry has no shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(u64::MAX) as usize)
            .collect();
        if &got != shape {
            bail!("{name}: checkpoint shape {got:?} but this model needs {shape:?}");
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|o| o.len() == 2)
            .with_context(|| format!("{name}: header entry has no data_offsets pair"))?;
        let start = offsets[0].as_u64().unwrap_or(u64::MAX);
        let end = offsets[1].as_u64().unwrap_or(0);
        let count: usize = shape.iter().product();
        if end < start || end - start != (count * width) as u64 {
            bail!("{name}: data_offsets span {start}..{end} does not hold {count} {dtype} values");
        }
        let mut bytes = vec![0u8; (end - start) as usize];
        file.seek(SeekFrom::Start(base + start))
            .with_context(|| format!("{name}: seeking in {}", path.display()))?;
        file.read_exact(&mut bytes)
            .with_context(|| format!("{name}: truncated payload in {}", path.display()))?;
        into.insert(name.clone(), widen(name, dtype, &bytes)?);
    }
    Ok(())
}

/// Read the vision tower + aligner out of an HF checkpoint directory (BF16 → f32), validating
/// every tensor's dtype and shape against `d` and failing fast with the tensor's name.
pub fn load_weights_from_safetensors(dir: &Path, d: &DsVitDims) -> Result<DsVitWeights> {
    // One open per shard, reading only the tensors it owns.
    let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
    for (shard, wanted) in group_by_shard(dir, required_tensors(d))? {
        read_shard(&shard, &wanted, &mut raw)?;
    }

    let mut take = |name: &str| -> Result<Vec<f32>> {
        raw.remove(name)
            .with_context(|| format!("{name}: missing after the shard read"))
    };
    let mut blocks = Vec::with_capacity(d.layers);
    for i in 0..d.layers {
        blocks.push(DsVitBlockWeights {
            norm1: take(&format!("vision.blocks.{i}.norm1.weight"))?,
            wqkv_w: take(&format!("vision.blocks.{i}.attn.wqkv.weight"))?,
            wqkv_b: take(&format!("vision.blocks.{i}.attn.wqkv.bias"))?,
            wo_w: take(&format!("vision.blocks.{i}.attn.wo.weight"))?,
            wo_b: take(&format!("vision.blocks.{i}.attn.wo.bias"))?,
            norm2: take(&format!("vision.blocks.{i}.norm2.weight"))?,
            w1: take(&format!("vision.blocks.{i}.mlp.w1.weight"))?,
            w2: take(&format!("vision.blocks.{i}.mlp.w2.weight"))?,
        });
    }
    Ok(DsVitWeights {
        patch_w: take("vision.patch_embed.proj.weight")?,
        patch_b: take("vision.patch_embed.proj.bias")?,
        blocks,
        norm: take("vision.norm.weight")?,
        aligner_w1: take("aligner.w1.weight")?,
        aligner_b1: take("aligner.w1.bias")?,
        aligner_w2: take("aligner.w2.weight")?,
        aligner_b2: take("aligner.w2.bias")?,
    })
}
