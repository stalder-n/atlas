// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE routing kernels for DeepSeek-V4-Flash-Vision-Exp — VL twins of
// `common/moe_topk_sqrtsoftplus.cu::moe_topk_sqrtsoftplus_batched` and
// `common/moe_hash_route.cu::moe_hash_route_batched`.
//
// WHAT IS DIFFERENT. The Vision-Exp checkpoint carries a SECOND expert
// selection bias, `e_score_correction_bias_vl`, used for IMAGE tokens only.
// A token is an image token exactly when its id is a sentinel, i.e.
// `token_ids[t] >= vocab_size` — sentinels travel in the token stream above
// the vocabulary and can never be sampled, because `lm_head` is [vocab_size].
// So:
//   * sqrtsoftplus router — selection bias is `bias_vl` for image tokens and
//     `bias` for text tokens; the WEIGHTS still come from the pre-bias
//     sqrtsoftplus scores, exactly as today.
//   * hash router — image tokens have no entry in the frozen `tid2eid` table
//     (it is indexed by vocabulary id), so they fall back to a LEARNED top-k
//     of `sqrtsoftplus(logits) + bias_vl`, with the same pre-bias weights,
//     `norm_topk_prob` normalisation and `routed_scaling_factor` scaling the
//     text path uses. Text tokens run the verbatim hash body.
//
// Both kernels are byte-identical to their `common/` twin for a text token.
// If `bias_vl == nullptr` BOTH degrade to the text path, so a launch-order bug
// reproduces today's behaviour instead of dereferencing null.
//
// THESE ARE COPIES. `common/` is shared by ~30 targets and is never edited for
// one model. If `common/moe_topk_sqrtsoftplus.cu` or `common/moe_hash_route.cu`
// changes, these two must be RE-DERIVED from them in the same change.
//
// LAUNCH CONTRACT — block MUST be (256,1,1). `num_warps` below is
// `BLOCK_SIZE / 32` with BLOCK_SIZE a compile-time 256, inherited verbatim
// from the common kernels, so a different block shape silently reduces
// (or over-reads) the cross-warp top-k reduction. Today's launchers
// (`ops/moe_grouped_a/topk.rs`, `ops/moe_gate.rs`) both use grid (N,1,1),
// block (256,1,1); keep it that way.
//
// Grid: (N, 1, 1)   Block: (256, 1, 1)  — one block per token.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32

// ── Learned top-k over sqrtsoftplus scores, selection biased by `sel_bias` ──
//
// The body of `common/moe_topk_sqrtsoftplus.cu`, taking one token's already
// offset rows. Both kernels below run it: the sqrtsoftplus router always, the
// hash router on its image arm. Whole-block, and every `__syncthreads()` here
// is reached by all 256 threads because both call sites are block-uniform.
__device__ void vl_learned_topk(
    const __nv_bfloat16* __restrict__ my_gate,   // [num_experts] this token's logits
    const float* __restrict__ sel_bias,          // [num_experts] selection bias
    unsigned int* __restrict__ my_indices,       // [top_k] output
    float* __restrict__ my_weights,              // [top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    __shared__ float s_score[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float logit = __bfloat162float(my_gate[i]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        s_score[i] = score;                    // pre-bias: this is what WEIGHTS use
        s_selection[i] = score + sel_bias[i];  // per-modality bias: selection only
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_score[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_selection[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_selection[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_score[idx];
            s_top_vals[t] = w;
            topk_sum += w;
        }
        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            my_weights[t] = s_top_vals[t] * scaling_factor;
        }
    }
}

// ── Learned sqrtsoftplus top-k with a per-modality selection bias ──
extern "C" __global__ void moe_topk_sqrtsoftplus_batched_vl(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts] BF16
    const float* __restrict__ bias,                  // [num_experts] F32 (text)
    unsigned int* __restrict__ expert_indices,       // [N, top_k] output
    float* __restrict__ expert_weights,              // [N, top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor,
    const float* __restrict__ bias_vl,               // [num_experts] F32 (image), may be null
    const unsigned int* __restrict__ token_ids,      // [N] u32 (device)
    unsigned int vocab_size                          // sentinel ids are >= this
) {
    const unsigned int token = blockIdx.x;

    // Block-uniform: every thread of this block serves the same token.
    const unsigned int tok = token_ids[token];
    const float* __restrict__ sel_bias =
        (bias_vl != nullptr && tok >= vocab_size) ? bias_vl : bias;

    vl_learned_topk(
        gate_logits + token * num_experts,
        sel_bias,
        expert_indices + token * top_k,
        expert_weights + token * top_k,
        num_experts, top_k, normalize, scaling_factor
    );
}

// ── Hash routing for text, learned VL routing for image tokens ──
//
// The two arms are selected by a BLOCK-UNIFORM predicate (`token_ids[blockIdx.x]`
// and a null check on a kernel argument), so the `__syncthreads()` calls inside
// `vl_learned_topk` are reached by every thread of the block that gets there.
extern "C" __global__ void moe_hash_route_batched_vl(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts] BF16
    const long* __restrict__ tid2eid,               // [vocab_size, top_k] i64
    const unsigned int* __restrict__ token_ids,     // [N] u32 (device)
    unsigned int* __restrict__ expert_indices,      // [N, top_k] output
    float* __restrict__ expert_weights,             // [N, top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor,
    const float* __restrict__ bias_vl,              // [num_experts] F32 (image), may be null
    unsigned int vocab_size                         // sentinel ids are >= this
) {
    const unsigned int token = blockIdx.x;
    const unsigned int tok = token_ids[token];

    const __nv_bfloat16* my_gate = gate_logits + (size_t)token * num_experts;
    unsigned int* my_indices = expert_indices + (size_t)token * top_k;
    float* my_weights = expert_weights + (size_t)token * top_k;

    if (tok >= vocab_size && bias_vl != nullptr) {
        // ── IMAGE ARM: learned top-k with `bias_vl` as the selection bias.
        //    `tid2eid` is unused: it is indexed by vocabulary id and a
        //    sentinel has no row in it. ──
        vl_learned_topk(my_gate, bias_vl, my_indices, my_weights,
                        num_experts, top_k, normalize, scaling_factor);
        return;
    }

    // ── TEXT ARM: verbatim moe_hash_route_batched ──
    if (threadIdx.x != 0) return;

    const long* row = tid2eid + (size_t)tok * (size_t)top_k;

    float w_local[MAX_TOP_K];
    unsigned int idx_local[MAX_TOP_K];
    float topk_sum = 0.0f;

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        unsigned int e = (unsigned int)row[t];
        if (e >= num_experts) e = 0;
        idx_local[t] = e;
        float logit = __bfloat162float(my_gate[e]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        w_local[t] = score;
        topk_sum += score;
    }

    if (normalize && topk_sum > 1e-20f) {
        for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
            w_local[t] /= topk_sum;
        }
    }

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        my_indices[t] = idx_local[t];
        my_weights[t] = w_local[t] * scaling_factor;
    }
}
