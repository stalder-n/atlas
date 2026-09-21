// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Spark — DeepSeek-V4-Flash-Vision-Exp ViT / aligner element ops.
//
// Reference: DeepSeek's `inference/vision.py` (pinned revision
// 86f746b36186f0e567729a5c06a8c918caba82a9) — the 3x3 pixel-unshuffle that
// feeds the aligner, the ViT MLP activation, and the aligner's SwiGLU.
//
// These three ops are the ONLY places the DeepSeek tower differs elementwise
// from the Qwen tower in `vision_encoder.cu` (copied verbatim into this
// target). Everything else — GEMMs, LayerNorm, RoPE/SDPA, the f32->bf16
// copies — is shared with that file.
//
// BF16 storage, f32 accumulation, `__restrict__` on every pointer, no `-DHDIM`
// dependence. Each kernel runs once (unshuffle/SwiGLU) or once per ViT block
// (GELU) per image, so simplicity beats peak bandwidth here — same posture as
// `vision_encoder.cu`.

#include <cuda_bf16.h>
#include <math.h>

#define SQRT2F 1.41421356237f

// ── 3x3 pixel unshuffle: [n_h*n_w, dim] -> [n_llm_h*n_llm_w, dim*9] ──
//
//   out[(oh*n_llm_w + ow)][c*9 + kh*3 + kw]
//       = (oh*3+kh < n_h && ow*3+kw < n_w)
//           ? x[(oh*3+kh)*n_w + (ow*3+kw)][c]
//           : 0
//
// Zero-padding (rather than a clamped read) is the whole reason this kernel
// exists: the ViT grid `n_h x n_w` is NOT required to be a multiple of 3, and
// the reference pads the missing patches with zeros before the aligner.
//
// Geometry mirrors `vision_encoder.cu::vision_spatial_merge`: one block per
// OUTPUT token, a strided thread loop over that token's `dim*9` features.
// Grid: (n_llm_h * n_llm_w, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void ds_vision_unshuffle3(
    const __nv_bfloat16* __restrict__ x,    // [n_h * n_w, dim]
    __nv_bfloat16* __restrict__ out,        // [n_llm_h * n_llm_w, dim * 9]
    unsigned int n_h, unsigned int n_w,
    unsigned int n_llm_h, unsigned int n_llm_w,
    unsigned int dim
) {
    unsigned int out_idx = blockIdx.x;
    if (out_idx >= n_llm_h * n_llm_w) return;

    const unsigned int oh = out_idx / n_llm_w;
    const unsigned int ow = out_idx % n_llm_w;
    const unsigned int width = dim * 9u;   // <= 1024*9 = 9216 for the Exp tower

    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        // ── the single most likely wrong-by-copy line in this file ──
        // The contract's feature layout is `c*9 + kh*3 + kw`, i.e. the CHANNEL
        // is the slow axis and the 3x3 tap is the fast one. That is the
        // TRANSPOSE of `vision_spatial_merge`, which decomposes its linear
        // index as `p_local = pi / D, d = pi % D` (patch slow, feature fast).
        // Do not "harmonise" the two.
        const unsigned int c  = i / 9u;
        const unsigned int k  = i % 9u;
        const unsigned int kh = k / 3u;
        const unsigned int kw = k % 3u;

        const unsigned int ph = oh * 3u + kh;
        const unsigned int pw = ow * 3u + kw;

        float v = 0.0f;
        if (ph < n_h && pw < n_w) {
            v = __bfloat162float(x[(size_t)(ph * n_w + pw) * dim + c]);
        }
        out[(size_t)out_idx * width + i] = __float2bfloat16(v);
    }
}

// ── Exact-erf GELU, in place ──
//
//   x[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
//
// DO NOT swap this for the tanh approximation `vision_encoder.cu::vision_gelu`
// uses. That kernel's own comment records why Qwen3-VL needs tanh: its vision
// config declares `hidden_act: "gelu_pytorch_tanh"`. DeepSeek's reference
// builds the ViT MLP with a bare `nn.GELU()`, whose default `approximate="none"`
// IS the erf form; the two differ by ~1e-4 per value and that error compounds
// across every ViT block.
//
// Written as a DIVISION by sqrt(2) rather than a multiply by its reciprocal, to
// mirror the reference expression exactly. `--fmad=false` is already set for
// this target in nvfp4/KERNEL.toml, so no contraction reassociates it.
//
// Grid: (ceil(n / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void ds_gelu_erf_inplace(
    __nv_bfloat16* __restrict__ x,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(0.5f * v * (1.0f + erff(v / SQRT2F)));
}

// ── SwiGLU over a fused gate_up tensor ──
//
//   gate = gate_up[:, :inter]        (torch `chunk(2, dim=-1)` order)
//   up   = gate_up[:, inter:]
//   out  = silu(gate) * up
//
// NO SWIGLU CLAMP. The 10.0 clamp in this target's `moe_silu_mul.cu` is the
// checkpoint's `swiglu_limit` for the MoE FFN; the ViT/aligner MLP in the
// reference is an unclamped `F.silu(gate) * up`. Adding the clamp here would
// apply a text-FFN config value to the vision tower.
//
// Geometry mirrors `moe_silu_mul.cu`: one thread per output element.
// `n * inter` <= 3456 * 2816 ~= 9.7e6 for the Exp tower, well inside u32.
// Grid: (ceil(n * inter / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void ds_swiglu_split(
    const __nv_bfloat16* __restrict__ gate_up,  // [n, 2 * inter]
    __nv_bfloat16* __restrict__ out,            // [n, inter]
    unsigned int n,
    unsigned int inter
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * inter) return;

    const unsigned int t = idx / inter;
    const unsigned int j = idx % inter;
    const size_t base = (size_t)t * 2u * (size_t)inter;

    float g = __bfloat162float(gate_up[base + j]);
    float u = __bfloat162float(gate_up[base + (size_t)inter + j]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    out[idx] = __float2bfloat16(g * sigmoid_g * u);
}
