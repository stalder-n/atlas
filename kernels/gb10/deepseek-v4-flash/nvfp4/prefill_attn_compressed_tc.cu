// SPDX-License-Identifier: AGPL-3.0-only
// V4-only compressed attention. Same 15-argument ABI as the scalar oracle.
// Grid (query_heads, ceil(seq_len/32), 1), block (128, 1, 1).
// BF16 WMMA computes 32 query rows x 16 keys, with FP32 QK accumulation.
// Both regions share one FP32 online softmax; probabilities and PV stay FP32
// (no BF16 probability rounding). The sink contributes only to the denominator.
#include <cuda_bf16.h>
#include <mma.h>

extern "C" __global__ void prefill_attn_compressed_tc(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ Kc,
    const __nv_bfloat16* __restrict__ Vc,
    const float* __restrict__ sinks,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int sliding_window,
    const float inv_sqrt_d
) {
    using namespace nvcuda;
    // Host selects the scalar kernel before launch for unsupported shapes.
    if (head_dim != 512 || num_kv_heads == 0 || ratio == 0 ||
        num_q_heads % num_kv_heads != 0 || blockDim.x != 128) return;
    const unsigned head = blockIdx.x;
    if (head >= num_q_heads) return;
    const unsigned tid = threadIdx.x, warp = tid / 32;
    const unsigned row0 = blockIdx.y * 32;
    const unsigned row = row0 + tid / 4, lane = tid % 4;
    const unsigned kvhead = head / (num_q_heads / num_kv_heads);
    const size_t qs = (size_t)num_q_heads * 512;
    const size_t ks = (size_t)num_kv_heads * 512;
    __shared__ __align__(32) __nv_bfloat16 qt[32 * 16];
    // Stored as [key, dim], interpreted as a column-major [dim, key] matrix.
    __shared__ __align__(32) __nv_bfloat16 kt[16 * 16];
    __shared__ __align__(32) float scores[32 * 16];
    float acc[128];
    #pragma unroll
    for (unsigned d = 0; d < 128; ++d) acc[d] = 0.0f;
    float maximum = -INFINITY, denominator = 0.0f;
    const unsigned raw_end = min(seq_len, row0 + 32);
    const unsigned raw_begin = sliding_window && row0 + 1 > sliding_window
        ? row0 + 1 - sliding_window : 0;
    const unsigned comp_end = min(n_comp, raw_end / ratio);

    for (unsigned region = 0; region < 2; ++region) {
        const unsigned begin = region == 0 ? raw_begin : 0;
        const unsigned end = region == 0 ? raw_end : comp_end;
        for (unsigned key0 = begin; key0 < end; key0 += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot;
            if (warp < 2) wmma::fill_fragment(dot, 0.0f);
            for (unsigned dim = 0; dim < 512; dim += 16) {
                for (unsigned i = tid; i < 32 * 16; i += 128) {
                    const unsigned qr = row0 + i / 16;
                    qt[i] = qr < seq_len ? Q[qr * qs + head * 512 + dim + i % 16]
                                               : __float2bfloat16(0.0f);
                }
                for (unsigned i = tid; i < 16 * 16; i += 128) {
                    const unsigned key = key0 + i / 16;
                    kt[i] = key < end
                        ? (region == 0 ? K[key * ks + kvhead * 512 + dim + i % 16]
                                       : Kc[(size_t)key * 512 + dim + i % 16])
                        : __float2bfloat16(0.0f);
                }
                __syncthreads();
                if (warp < 2) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                   __nv_bfloat16, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                   __nv_bfloat16, wmma::col_major> b;
                    wmma::load_matrix_sync(a, qt + warp * 16 * 16, 16);
                    wmma::load_matrix_sync(b, kt, 16);
                    wmma::mma_sync(dot, a, b, dot);
                }
                __syncthreads();
            }
            if (warp < 2)
                wmma::store_matrix_sync(scores + warp * 16 * 16, dot, 16, wmma::mem_row_major);
            __syncthreads();
            if (row < seq_len) {
                for (unsigned j = 0; j < 16 && key0 + j < end; ++j) {
                    const unsigned key = key0 + j;
                    const bool visible = region == 0
                        ? key <= row && (!sliding_window || row - key < sliding_window)
                        : key < (row + 1) / ratio;
                    if (!visible) continue;
                    const float score = scores[(tid / 4) * 16 + j] * inv_sqrt_d;
                    const float next_max = fmaxf(maximum, score);
                    const float old_scale = __expf(maximum - next_max);
                    const float probability = __expf(score - next_max);
                    const __nv_bfloat16* value = region == 0
                        ? V + key * ks + kvhead * 512 : Vc + (size_t)key * 512;
                    #pragma unroll
                    for (unsigned d = 0; d < 128; ++d)
                        acc[d] = acc[d] * old_scale + probability * __bfloat162float(value[lane + d * 4]);
                    denominator = denominator * old_scale + probability;
                    maximum = next_max;
                }
            }
            __syncthreads();
        }
    }
    if (row < seq_len) {
        if (sinks != nullptr) {
            const float sink = sinks[head];
            const float next_max = fmaxf(maximum, sink);
            const float old_scale = __expf(maximum - next_max);
            #pragma unroll
            for (unsigned d = 0; d < 128; ++d) acc[d] *= old_scale;
            denominator = denominator * old_scale + __expf(sink - next_max);
        }
        #pragma unroll
        for (unsigned d = 0; d < 128; ++d)
            O[row * qs + head * 512 + lane + d * 4] = __float2bfloat16(acc[d] / denominator);
    }
}
