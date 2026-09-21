// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4 CSA prefill attention, IMAGE-SPAN variant: identical to
// `prefill_attn_compressed.cu` except that the raw arm reads its key range
// from a PER-QUERY inclusive bounds pair instead of deriving it from the
// `(sliding_window, causal)` rule.
//
// WHY A SEPARATE KERNEL. A prompt that contains image sentinel blocks is not
// causal inside those blocks: every token of an image span sees every other
// token of the same span, while text stays causal. That is a per-query key
// interval, not a scalar window, so it cannot be expressed in the scalar
// kernel's `sliding_window` argument.
//
// The bounds come from `spark_model::vision_deepseek::visibility::window_bounds`
// and are INCLUSIVE: query `q` attends raw keys `win_lo[q] ..= win_hi[q]`, with
// `win_lo[q] <= win_hi[q] < seq_len` guaranteed by the host. For a pure-text
// query the host emits `lo = (q + 1 >= W) ? q + 1 - W : 0`, `hi = q`, which
// reproduces the scalar kernel exactly — hence no `sliding_window` argument.
//
// Everything else is the original, byte for byte: the compressed arm, the
// per-head attention sink, the online softmax, the epilogue.
//
// Q/K/V are [S, num_heads, head_dim] (num_kv_heads=1 → MQA broadcast).
// Compressed K/V are [n_comp, head_dim].
//
// Layout matches prefill_attn_compressed: 128 threads = 16 rows × 8 dim-lanes,
// each dim-lane owns 64 of the head_dim=512 output dims.
// Grid: (num_q_heads, ceil(S/16), batch)  Block: (128,1,1)
//
// WARP SAFETY. `q_row = q_block*16 + tid/8` is uniform across each aligned
// group of 8 lanes, and the `__shfl_xor_sync` masks below (1/2/4) only exchange
// inside that group, so trip counts stay uniform among actual shuffle partners
// even though different groups now walk different key ranges — the same
// property the scalar kernel relies on.

#include <cuda_bf16.h>

#define BR 16

extern "C" __global__ void prefill_attn_compressed_vis(
    const __nv_bfloat16* __restrict__ Q,       // [S, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K,       // [S, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V,       // [S, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ Kc,      // [n_comp, head_dim]  (kv head 0)
    const __nv_bfloat16* __restrict__ Vc,      // [n_comp, head_dim]
    const float* __restrict__ sinks,   // [num_q_heads]  per-head sink logit (FP32: checkpoint-native; reading as bf16 hard-zeroed 7 heads)
    __nv_bfloat16* __restrict__ O,             // [S, num_q_heads, head_dim]
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int* __restrict__ win_lo,   // [S] inclusive first raw key per query
    const unsigned int* __restrict__ win_hi,   // [S] inclusive last  raw key per query
    const float inv_sqrt_d
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (q_head >= num_q_heads) return;

    const unsigned int q_row = q_block * BR + (tid / 8);
    const bool valid = q_row < seq_len;
    const unsigned int dim_lane = tid % 8;
    const unsigned int dim_start = dim_lane * 64;
    if (dim_start >= head_dim) return;

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Qr = Q + (size_t)q_row * q_stride + (size_t)q_head * head_dim;

    float m = -1e30f, l = 0.0f;
    float o_acc[64];
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] = 0.0f;

    // helper: process one (K_row, V_row) key with online softmax
    #define ATTEND(KROW, VROW)                                                   \
    do {                                                                          \
        float dot = 0.0f;                                                         \
        if (valid) {                                                              \
            for (unsigned int d = dim_start; d < dim_start + 64 && d < head_dim; ++d) \
                dot += __bfloat162float(Qr[d]) * __bfloat162float((KROW)[d]);     \
        }                                                                         \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);                               \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);                               \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 4);                               \
        float score = dot * inv_sqrt_d;                                           \
        float m_new = fmaxf(m, score);                                            \
        float eo = __expf(m - m_new);                                            \
        float en = __expf(score - m_new);                                        \
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)         \
            o_acc[d] = o_acc[d] * eo + en * __bfloat162float((VROW)[dim_start + d]); \
        l = l * eo + en;                                                          \
        m = m_new;                                                               \
    } while (0)

    // ── raw keys (per-query INCLUSIVE bounds: win_lo[q_row] ..= win_hi[q_row]) ──
    // The 1u/0u sentinel makes the loop empty for an out-of-range row WITHOUT an
    // unsigned underflow and without ever indexing win_lo/win_hi out of bounds.
    unsigned int k_lo = 1u, k_hi = 0u;
    if (valid) {
        k_lo = win_lo[q_row];
        k_hi = win_hi[q_row];
        // Memory-safety guard (CWE-125) against a host bug, not a policy
        // default: the host owns `hi < seq_len` and never emits anything else.
        if (k_hi >= seq_len) k_hi = seq_len - 1u;
    }
    for (unsigned int kp = k_lo; kp <= k_hi; ++kp) {
        const __nv_bfloat16* Kr = K + (size_t)kp * kv_stride + (size_t)kv_head * head_dim;
        const __nv_bfloat16* Vr = V + (size_t)kp * kv_stride + (size_t)kv_head * head_dim;
        ATTEND(Kr, Vr);
    }

    // ── compressed keys (windowed-causal: window w visible if (w+1)*ratio <= q_row+1) ──
    unsigned int comp_vis = valid ? ((q_row + 1) / ratio) : 0;
    if (comp_vis > n_comp) comp_vis = n_comp;
    for (unsigned int w = 0; w < comp_vis; ++w) {
        const __nv_bfloat16* Kr = Kc + (size_t)w * head_dim;
        const __nv_bfloat16* Vr = Vc + (size_t)w * head_dim;
        ATTEND(Kr, Vr);
    }

    // ── attention sink: per-head logit in the denominator only (no value) ──
    if (valid && sinks != nullptr) {
        float sg = sinks[q_head];
        float m_new = fmaxf(m, sg);
        float eo = __expf(m - m_new);
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] *= eo;
        l = l * eo + __expf(sg - m_new);
        m = m_new;
    }

    if (valid) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* Or = O + (size_t)q_row * q_stride + (size_t)q_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
            Or[dim_start + d] = __float2bfloat16(o_acc[d] * inv_l);
    }
    #undef ATTEND
}
