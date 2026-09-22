// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the DeepSeek vision tower.
//!
//! No GPU and no numerics: [`MockGpuBackend`] records kernel lookups, launch
//! geometry and scalar arguments, and moves real bytes for the copies. What
//! CAN be pinned here is the wiring — which kernel, which module, which
//! grid/block, which scalars — plus the one host-computed input to the blind
//! kernels: the 2D RoPE tables, checked against U1.3's golden-backed
//! reference.

use avarok_core::config::{VisionConfig, VisionFlavor};
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepseekVisionEncoder, DsViTBlock};

const DIM: usize = 32;
const HEADS: usize = 2;
const HEAD_DIM: usize = DIM / HEADS;
const INTER: usize = 8;
const PATCH: usize = 2;
const DEPTH: usize = 2;
const OUT_DIM: usize = 16;
const MAX_N_TOKEN: usize = 4;
const THETA: f64 = 10_000.0;

fn tiny_config() -> VisionConfig {
    VisionConfig {
        depth: DEPTH,
        hidden_size: DIM,
        num_heads: HEADS,
        patch_size: PATCH,
        temporal_patch_size: 1,
        spatial_merge_size: 3,
        intermediate_size: INTER,
        out_hidden_size: OUT_DIM,
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: 0,
        video_pad_token_id: 0,
        max_pixels: None,
        flavor: VisionFlavor::DeepseekV4,
        rope_theta: THETA,
        max_n_token: MAX_N_TOKEN,
        min_pixels: 16,
        max_wh_ratio: 8,
    }
}

fn tiny_encoder(gpu: &MockGpuBackend) -> DeepseekVisionEncoder {
    let p = DevicePtr(0x1000);
    let blocks = (0..DEPTH)
        .map(|_| DsViTBlock {
            norm1_w: p,
            qkv_w: p,
            qkv_b: p,
            wo_w: p,
            wo_b: p,
            norm2_w: p,
            w1_w: p,
            w2_w: p,
        })
        .collect();
    DeepseekVisionEncoder::new(p, p, blocks, p, p, p, p, p, &tiny_config(), OUT_DIM, gpu)
        .expect("tiny encoder")
}

/// Scalar (non-pointer) arguments of a launch, as raw little-endian u32s.
fn scalars(l: &MockLaunch) -> Vec<u32> {
    l.args
        .iter()
        .filter_map(|a| match a {
            MockArg::Bytes(b) if b.len() == 4 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            _ => None,
        })
        .collect()
}

fn n_ptrs(l: &MockLaunch) -> usize {
    l.args
        .iter()
        .filter(|a| matches!(a, MockArg::Buffer(_)))
        .count()
}

#[test]
fn geometry_is_derived_from_the_vision_config() {
    let gpu = MockGpuBackend::new();
    let e = tiny_encoder(&gpu);
    assert_eq!(e.head_dim, HEAD_DIM);
    assert_eq!(e.patch_in(), 3 * PATCH * PATCH);
    // ds² · max_n_token — NOT a Qwen area bound.
    assert_eq!(e.p_max, 9 * MAX_N_TOKEN);
}

#[test]
fn every_kernel_is_looked_up_hard_and_the_norm_is_the_vanilla_one() {
    let gpu = MockGpuBackend::new();
    let _ = tiny_encoder(&gpu);
    let looked: Vec<(String, String)> = gpu.kernel_lookups_snapshot();
    let want = [
        ("rms_norm_vanilla", "rms_norm_vanilla"),
        ("gemm", "dense_gemm_bf16_pipelined"),
        ("gemm", "dense_gemm_bf16_f32out"),
        ("vision_encoder", "vision_add_bias"),
        ("vision_encoder", "vision_add_inplace"),
        ("vision_encoder", "vision_bf16_copy"),
        ("vision_encoder", "vision_f32_to_bf16"),
        ("vision_encoder", "vit_rope_deinterleave"),
        ("vision_encoder", "vit_softmax_rows"),
        ("vision_encoder", "vit_scatter_head"),
        ("vision_deepseek", "ds_vision_unshuffle3"),
        ("vision_deepseek", "ds_gelu_erf_inplace"),
        ("vision_deepseek", "ds_swiglu_split"),
    ];
    for (m, k) in want {
        assert!(
            looked.iter().any(|(lm, lk)| lm == m && lk == k),
            "missing kernel lookup {m}::{k}; got {looked:?}"
        );
    }
    assert_eq!(
        looked.len(),
        want.len(),
        "unexpected extra lookups: {looked:?}"
    );
    // The offset-from-1 twin would add 1.0 to every ViT norm weight.
    assert!(
        !looked.iter().any(|(m, k)| m == "norm" && k == "rms_norm"),
        "the DS ViT must not bind norm::rms_norm"
    );
}

#[test]
fn a_denied_kernel_fails_the_build_rather_than_yielding_a_null_handle() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("vision_deepseek", "ds_swiglu_split");
    let p = DevicePtr(0x1000);
    let blocks = (0..DEPTH)
        .map(|_| DsViTBlock {
            norm1_w: p,
            qkv_w: p,
            qkv_b: p,
            wo_w: p,
            wo_b: p,
            norm2_w: p,
            w1_w: p,
            w2_w: p,
        })
        .collect();
    let err =
        DeepseekVisionEncoder::new(p, p, blocks, p, p, p, p, p, &tiny_config(), OUT_DIM, &gpu)
            .err()
            .expect("absent vision kernel must fail the build");
    assert!(format!("{err:#}").contains("ds_swiglu_split"), "{err:#}");
}

#[test]
fn a_qwen_flavored_config_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut v = tiny_config();
    v.flavor = VisionFlavor::Qwen;
    let p = DevicePtr(0x1000);
    let err = DeepseekVisionEncoder::new(p, p, Vec::new(), p, p, p, p, p, &v, OUT_DIM, &gpu)
        .err()
        .expect("Qwen config must be refused");
    assert!(format!("{err:#}").contains("vision config"), "{err:#}");
}

#[test]
fn rope_tables_match_the_host_reference_with_duplicated_halves() {
    let gpu = MockGpuBackend::new();
    let e = tiny_encoder(&gpu);
    let (n_h, n_w) = (4usize, 5usize);
    let p = n_h * n_w;
    let cos_d = gpu.alloc(p * HEAD_DIM * 2).unwrap();
    let sin_d = gpu.alloc(p * HEAD_DIM * 2).unwrap();
    e.build_rope_cossin_into(n_h, n_w, cos_d, sin_d, &gpu, 0)
        .unwrap();

    let read = |ptr: DevicePtr| -> Vec<f32> {
        let mut b = vec![0u8; p * HEAD_DIM * 2];
        gpu.copy_d2h(ptr, &mut b).unwrap();
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    let (got_cos, got_sin) = (read(cos_d), read(sin_d));

    let rope_dim = HEAD_DIM / 2;
    let (ref_cos, ref_sin) = crate::vision_deepseek_ref::rope_cos_sin(n_h, n_w, rope_dim, THETA);
    assert_eq!(ref_cos.len(), p * rope_dim);
    for t in 0..p {
        for j in 0..rope_dim {
            for half in [0usize, rope_dim] {
                let g = got_cos[t * HEAD_DIM + half + j];
                let r = ref_cos[t * rope_dim + j];
                assert!((g - r).abs() < 1e-2, "cos[{t}][{}] {g} vs {r}", half + j);
                let g = got_sin[t * HEAD_DIM + half + j];
                let r = ref_sin[t * rope_dim + j];
                assert!((g - r).abs() < 1e-2, "sin[{t}][{}] {g} vs {r}", half + j);
            }
        }
    }
    // Row-major: patch 1 is (h=0, w=1), so its FIRST block (height angles)
    // must still be the h=0 row while its second block has moved.
    assert!(
        (got_cos[HEAD_DIM] - 1.0).abs() < 1e-3,
        "h angle at (0,1) is not 0"
    );
    assert!(
        got_sin[HEAD_DIM + rope_dim / 2] > 0.0,
        "w angle at (0,1) did not advance"
    );
}

#[test]
fn encode_image_refuses_a_grid_past_capacity_and_a_short_pixel_buffer() {
    let gpu = MockGpuBackend::new();
    let e = tiny_encoder(&gpu);
    let dst = gpu.alloc(MAX_N_TOKEN * OUT_DIM * 2).unwrap();

    // 7×7 = 49 patches against p_max = 36.
    let px = vec![0.0f32; 49 * e.patch_in()];
    let err = e
        .encode_image(&px, 7, 7, dst, &gpu, 0, 0)
        .expect_err("oversized grid must be refused");
    assert!(format!("{err:#}").contains("holds 36"), "{err:#}");

    let err = e
        .encode_image(&[0.0f32; 4], 4, 5, dst, &gpu, 0, 0)
        .expect_err("short pixel buffer must be refused");
    assert!(format!("{err:#}").contains("pixel values"), "{err:#}");
}

/// The whole per-image kernel stream, pinned by geometry and scalar args.
#[test]
fn encode_image_emits_the_reference_pipeline() {
    let gpu = MockGpuBackend::new();
    let e = tiny_encoder(&gpu);
    let (n_h, n_w) = (4usize, 5usize);
    let p = n_h * n_w;
    let dst = gpu.alloc(MAX_N_TOKEN * OUT_DIM * 2).unwrap();
    let px = vec![0.25f32; p * e.patch_in()];

    let grid = e.encode_image(&px, n_h, n_w, dst, &gpu, 7, 3).unwrap();
    // ceil(4/3), ceil(5/3) — `geometry::n_llm`, not a local div_ceil.
    assert_eq!(grid, (2, 2));

    let ls = gpu.launches_snapshot();
    assert!(
        ls.iter().all(|l| l.stream == 7),
        "every launch rides the caller's stream"
    );

    // RMSNorm: one per attention sub-block, one per MLP sub-block, one final.
    let rms: Vec<&MockLaunch> = ls
        .iter()
        .filter(|l| n_ptrs(l) == 3 && l.block == [DIM as u32, 1, 1] && l.grid == [p as u32, 1, 1])
        .collect();
    assert_eq!(rms.len(), 2 * DEPTH + 1, "RMSNorm launch count");
    for l in &rms {
        let s = scalars(l);
        assert_eq!(s[0], DIM as u32, "RMSNorm hidden_size");
        assert_eq!(
            f32::from_bits(s[1]),
            1e-6,
            "the ViT's own eps, not the LLM's"
        );
    }

    // Attention softmax: one per (block, head), with D = head_dim — the folded
    // 1/√head_dim that IS the reference's SDPA scale.
    let softmax: Vec<&MockLaunch> = ls
        .iter()
        .filter(|l| n_ptrs(l) == 2 && scalars(l) == [p as u32, HEAD_DIM as u32])
        .collect();
    assert_eq!(
        softmax.len(),
        DEPTH * HEADS,
        "one softmax per head per block"
    );

    // SwiGLU split: [p, inter], once per block.
    let swiglu: Vec<&MockLaunch> = ls
        .iter()
        .filter(|l| n_ptrs(l) == 2 && scalars(l) == [p as u32, INTER as u32])
        .collect();
    assert_eq!(swiglu.len(), DEPTH, "one SwiGLU per block");

    // The unshuffle: one block per output token, (n_h, n_w, n_llm_h, n_llm_w, dim).
    let un: Vec<(usize, &MockLaunch)> = ls
        .iter()
        .enumerate()
        .filter(|(_, l)| scalars(l) == [n_h as u32, n_w as u32, 2, 2, DIM as u32])
        .collect();
    assert_eq!(un.len(), 1, "exactly one 3x3 unshuffle");
    let (un_idx, un_l) = un[0];
    assert_eq!(un_l.grid, [4, 1, 1]);
    assert_eq!(un_l.block, [256, 1, 1]);

    // The aligner's GELU is the erf kernel over n_tok·out_dim, after the
    // unshuffle and before the last GEMM.
    let gelu: Vec<(usize, &MockLaunch)> = ls
        .iter()
        .enumerate()
        .filter(|(_, l)| n_ptrs(l) == 1 && scalars(l) == [(4 * OUT_DIM) as u32])
        .collect();
    assert_eq!(gelu.len(), 1, "exactly one aligner GELU");
    assert!(gelu[0].0 > un_idx, "GELU must follow the unshuffle");

    // Last launch is the bias add of aligner.w2 over [n_tok, out_dim].
    let last = ls.last().unwrap();
    assert_eq!(scalars(last), [4, OUT_DIM as u32], "aligner w2 bias add");
}
