// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the pieces of native DSpark that do not need a GPU.
//!
//! The 3-stage numeric path (mHC → MLA-over-a-ring → 256-expert MoE, ×3) is
//! verifiable only on the Spark. What IS verifiable here is the part Atlas
//! re-implements AROUND it: the greedy Markov chain and the two-slice
//! confidence dot, fed the golden's OWN `x` and pre-Markov logits so the stage
//! bodies are out of the picture; the ring's index algebra against the ring
//! the reference actually built; and the "inert when disabled" guard.
//!
//! The golden (`fixtures/dspark_tiny_golden.json`) is
//! `Transformer.forward` + `forward_spec` from DeepSeek's own
//! `inference/model.py` at the pinned revision, on a tiny config — see
//! `gen_dspark_tiny_golden.py` beside it, and the fixture's `_generator`
//! block for the two recorded substitutions.

use serde_json::Value;

use super::markov_confidence::MarkovChainRef;
use super::ring;

const GOLDEN: &str = include_str!("fixtures/dspark_tiny_golden.json");

fn golden() -> Value {
    serde_json::from_str(GOLDEN).expect("dspark_tiny_golden.json is not valid JSON")
}

fn cfg_usize(g: &Value, key: &str) -> usize {
    g["config"][key]
        .as_u64()
        .unwrap_or_else(|| panic!("config.{key}")) as usize
}

/// A capture's `data`, which is the whole tensor when `stride == 1`.
fn full(g: &Value, name: &str) -> Vec<f32> {
    let c = &g["captures"][name];
    assert_eq!(
        c["stride"].as_u64().unwrap(),
        1,
        "{name} is strided; the host reference needs it whole"
    );
    c["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn ints(g: &Value, name: &str) -> Vec<u32> {
    g["captures"][name]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as u32)
        .collect()
}

// ── the generator's deterministic weight RNG, reproduced bit for bit ──
//
// `fixed(name, i)`: z = splitmix64(fnv1a64(name) ^ (i * GOLD));
//                   u = (z >> 40) / 2^24;  value = f32((2u - 1) * scale + offset)
const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(GOLD);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fixed(name: &str, n: usize, scale: f64, offset: f64) -> Vec<f32> {
    let salt = fnv1a64(name);
    (0..n)
        .map(|i| {
            let z = splitmix64(salt ^ (i as u64).wrapping_mul(GOLD));
            let u = (z >> 40) as f64 / (1u64 << 24) as f64;
            ((2.0 * u - 1.0) * scale + offset) as f32
        })
        .collect()
}

/// Round to BF16 and back — the generator stores `markov_w1` in the model's
/// default dtype (bfloat16), and the chain's bigram embedding is that value,
/// not the f32 the RNG produced.
fn to_bf16(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|&x| {
            let bits = x.to_bits();
            // round-to-nearest-even on the low 16 bits, as torch's f32→bf16 does
            let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            f32::from_bits(rounded & 0xFFFF_0000)
        })
        .collect()
}

fn ck(v: &[f32]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, &x)| x as f64 * (i as f64 + 1.0))
        .sum()
}

fn weight(g: &Value, name: &str) -> Vec<f32> {
    let meta = &g["weights_meta"][name];
    let n = meta["n"].as_u64().unwrap() as usize;
    let scale = meta["scale"].as_f64().unwrap();
    let offset = meta["offset"].as_f64().unwrap();
    let v = fixed(name, n, scale, offset);
    let v = if meta["dtype"].as_str().unwrap() == "bfloat16" {
        to_bf16(&v)
    } else {
        v
    };
    let want = meta["ck"].as_f64().unwrap();
    let got = ck(&v);
    assert!(
        (got - want).abs() <= want.abs() * 1e-9 + 1e-6,
        "{name}: reproduced checksum {got} != generator's {want} — the Rust RNG has drifted \
         from `gen_dspark_tiny_golden.py::fixed`"
    );
    v
}

/// 🔴 THE unit test of this file: the γ-row greedy chain and the confidence
/// dot, against DeepSeek's own outputs, fed the golden's own inputs.
///
/// It pins all three things `dflash_head/markov.rs` gets differently:
/// row 0 IS biased (by the anchor), the confidence reads the PRE-`norm` `x`,
/// and there is no bias term.
#[test]
fn the_markov_chain_and_confidence_match_the_reference() {
    let g = golden();
    let gamma = cfg_usize(&g, "dspark_block_size");
    let vocab = cfg_usize(&g, "vocab_size");
    let rank = cfg_usize(&g, "dspark_markov_rank");
    let hidden = cfg_usize(&g, "dim");

    let w1 = weight(&g, "mtp.2.markov_head.markov_w1.weight");
    let w2 = weight(&g, "mtp.2.markov_head.markov_w2.weight");
    let conf_w = weight(&g, "mtp.2.confidence_head.proj.weight");
    assert_eq!(conf_w.len(), hidden + rank);

    let logits = full(&g, "logits_pre_markov");
    let x = full(&g, "x_pre_norm");
    let want_ids = ints(&g, "output_ids");
    assert_eq!(want_ids.len(), gamma + 1);

    let out = MarkovChainRef {
        logits: &logits,
        markov_w1: &w1,
        markov_w2: &w2,
        x: &x,
        conf_w: &conf_w,
        gamma,
        vocab,
        rank,
        hidden,
    }
    .run(want_ids[0]);

    assert_eq!(
        out.output_ids, want_ids,
        "greedy chain diverged from the reference"
    );

    // The bigram embedding the chain looked up must be the one the reference
    // fed its confidence head — this is what proves row i is conditioned on
    // `output_ids[i]` and not on `output_ids[i+1]`.
    let want_embed = full(&g, "markov_embed");
    for i in 0..gamma {
        let prev = out.output_ids[i] as usize;
        for r in 0..rank {
            let got = w1[prev * rank + r];
            let want = want_embed[i * rank + r];
            assert!(
                (got - want).abs() <= 1e-6,
                "markov_embed[{i}][{r}]: {got} != {want}"
            );
        }
    }

    let want_conf = full(&g, "confidence");
    for (i, (&got, &want)) in out.confidence.iter().zip(&want_conf).enumerate() {
        assert!(
            (got - want).abs() <= 2e-4 * want.abs().max(1.0),
            "confidence[{i}]: {got} != {want}"
        );
    }

    // Whole-tensor agreement on the biased logits, via the generator's own
    // index-weighted checksum (the recorded sample is strided).
    let want_ck = g["captures"]["logits_post_markov"]["ck"].as_f64().unwrap();
    let got_ck = ck(&out.biased_logits);
    assert!(
        (got_ck - want_ck).abs() <= want_ck.abs() * 1e-4 + 1e-3,
        "biased logits checksum {got_ck} != {want_ck}"
    );
}

/// The chain must be a CHAIN: perturbing the anchor has to change the drafts.
/// Without this, a bug that ignored `output_ids[i]` and always biased by the
/// anchor would still pass the golden comparison above on a short vocabulary.
#[test]
fn the_chain_is_conditioned_on_its_own_previous_token() {
    let g = golden();
    let gamma = cfg_usize(&g, "dspark_block_size");
    let vocab = cfg_usize(&g, "vocab_size");
    let rank = cfg_usize(&g, "dspark_markov_rank");
    let hidden = cfg_usize(&g, "dim");
    let w1 = weight(&g, "mtp.2.markov_head.markov_w1.weight");
    let w2 = weight(&g, "mtp.2.markov_head.markov_w2.weight");
    let conf_w = weight(&g, "mtp.2.confidence_head.proj.weight");
    let logits = full(&g, "logits_pre_markov");
    let x = full(&g, "x_pre_norm");
    let anchor = ints(&g, "output_ids")[0];

    let chain = MarkovChainRef {
        logits: &logits,
        markov_w1: &w1,
        markov_w2: &w2,
        x: &x,
        conf_w: &conf_w,
        gamma,
        vocab,
        rank,
        hidden,
    };
    let base = chain.run(anchor);
    let other = chain.run((anchor + 1) % vocab as u32);
    assert_ne!(
        base.output_ids, other.output_ids,
        "a different anchor produced the same chain — row 0 is not biased"
    );
    assert_ne!(
        base.confidence, other.confidence,
        "confidence is independent of the chain — the markov_embed slice is not being read"
    );
}

/// The reference's prefill `cutoff` split IS `pos % window`. The golden's
/// prompt is 11 tokens against a window of 8, so it wraps.
#[test]
fn prefill_drain_reproduces_the_reference_cutoff_split() {
    let g = golden();
    let window = cfg_usize(&g, "window_size");
    let prompt = cfg_usize(&g, "prompt_len");
    assert!(prompt > window, "the golden must exercise the wrap");

    // `main_kv[:, -win:].split([win - cutoff, cutoff])` with `cutoff = P % win`
    // writes the first piece at `kv_cache[:, cutoff:win]` and the second at
    // `kv_cache[:, :cutoff]`.
    let cutoff = prompt % window;
    let mut expected: Vec<(usize, usize)> = Vec::new();
    for (i, pos) in (prompt - window..prompt).enumerate() {
        let slot = if i < window - cutoff {
            cutoff + i
        } else {
            i - (window - cutoff)
        };
        expected.push((pos, slot));
    }
    assert_eq!(ring::prefill_drain(prompt, window), expected);
}

/// After the prompt the ring is full and the first propose drafts from
/// `position = prompt_len`, which is the golden's decode step.
#[test]
fn the_golden_decode_step_leaves_a_full_ring_and_five_block_keys() {
    let g = golden();
    let window = cfg_usize(&g, "window_size");
    let block = cfg_usize(&g, "dspark_block_size");
    let prompt = cfg_usize(&g, "prompt_len");

    let len = ring::ring_len(prompt, window);
    assert_eq!(len, window, "an 11-token prompt fills an 8-slot ring");
    let (lo, hi) = ring::key_bounds(len, block);
    assert_eq!((lo, hi), (0, (window + block - 1) as u32));
    assert_eq!(ring::block_base(len), window);
    assert_eq!(ring::ring_slots(window, block), window + block);

    // The reference's ring capture confirms the depth the bounds assume.
    let shape = g["captures"]["stage0.ring_after_prefill"]["shape"]
        .as_array()
        .unwrap();
    assert_eq!(shape[1].as_u64().unwrap() as usize, window);
}

/// 🔴 Inert when disabled. `load_v4_dspark_module` must return `None` — never
/// an error, never a partial load — on a text checkpoint, so the 0731 text
/// path and every model in the support matrix are byte-for-byte unaffected.
#[test]
fn the_loader_is_inert_without_native_stages() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
    use std::collections::HashMap;

    let gpu = MockGpuBackend::new();
    let empty = WeightStore::from_map(HashMap::new());
    let mut config = avarok_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();

    // (a) `dspark_block_size == 0` — a pre-DSpark checkpoint.
    config.dspark_block_size = 0;
    assert!(
        crate::weight_loader::deepseek_v4::load_v4_dspark_module(&empty, &config, &gpu, &[])
            .unwrap()
            .is_none()
    );

    // (b) `dspark_block_size > 0` but no `mtp.0.main_proj.weight` — the LEGACY
    //     single-module MTP family, which `load_v4_mtp_module` owns.
    config.dspark_block_size = 5;
    let legacy = WeightStore::from_map(HashMap::from([(
        "mtp.0.enorm.weight".to_string(),
        WeightTensor {
            ptr: spark_runtime::gpu::DevicePtr::NULL,
            shape: vec![4096],
            dtype: WeightDtype::BF16,
        },
    )]));
    assert!(
        crate::weight_loader::deepseek_v4::load_v4_dspark_module(&legacy, &config, &gpu, &[])
            .unwrap()
            .is_none()
    );
}

/// The capture band's addressing, at the shapes the golden uses.
#[test]
fn the_capture_band_row_is_the_targets_concatenated() {
    let g = golden();
    let hidden = cfg_usize(&g, "dim");
    let n_targets = g["config"]["dspark_target_layer_ids"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        super::row_stride_elems(n_targets, hidden),
        n_targets * hidden
    );
    // …and that stride IS the width the reference's `main_proj` consumes.
    let captured = g["captures"]["main_hidden_in"]["shape"].as_array().unwrap();
    assert_eq!(
        captured[2].as_u64().unwrap() as usize,
        super::row_stride_elems(n_targets, hidden)
    );
    for slot in 0..n_targets {
        assert_eq!(super::slot_offset_elems(slot, hidden), slot * hidden);
        assert_eq!(super::band_row(0, slot, n_targets, n_targets), Some(slot));
    }
}
