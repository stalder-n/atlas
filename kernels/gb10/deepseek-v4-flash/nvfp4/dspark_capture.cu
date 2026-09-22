// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4 native DSpark target capture: the unweighted MEAN over the
// manifold-constrained hyper-connection (mHC) streams of one post-layer
// residual, scattered into a `[rows, n_targets * hidden]` capture band.
//
// WHY THIS EXISTS. `Transformer.forward` in the reference
// (`inference/model.py`, l. 986-1012) accumulates the DSpark target hiddens as
//
//     for i, layer in enumerate(self.layers):
//         h = layer(...)
//         if i in self.dspark_target_layer_ids:
//             main_hiddens.append(h.mean(dim=2))     # mean over the hc streams
//     main_hidden = torch.cat(main_hiddens, dim=-1)  # [.., n_targets * dim]
//
// `h` is the FP32 highway `[T, hc_mult, hidden]` that `hyper_connection.cu`
// maintains; the mean over `dim=2` is over hc_mult. The learned `hc_head` in
// that same file is the WEIGHTED (Sinkhorn) collapse and is NOT this: it reads
// `head_fn`/`head_base`/`head_scale`, which the target layers do not own at
// layers 40/41/42. The GLM target ships an unweighted `glm5next_hc_head`, but
// it lives in the `glm5next_mhc` module of a different model target and must
// not be reached for from here.
//
// BATCHED ON PURPOSE. One launch per (capture layer, prefill chunk): the
// DFlash capture it mirrors issues `proc_count` strided `copy_d2d_async`
// calls, which at a 2048-token prefill x 3 capture layers is ~6k launches.
//
// Destination addressing is in ELEMENTS (not bytes) so the caller can express
// both band layouts with one kernel:
//   dst[r * dst_row_stride_elems + dst_slot_off_elems + d]
// with `dst_row_stride_elems = n_targets * hidden` and
// `dst_slot_off_elems = slot * hidden` for the `slot`-th capture layer.

#include <cuda_bf16.h>

#define DSPARK_CAPTURE_BLOCK 256

extern "C" __global__ void dspark_hc_mean_capture(
    const float* __restrict__ streams,   // [*, hc_mult, hidden] FP32 mHC highway
    __nv_bfloat16* __restrict__ dst,     // [rows, dst_row_stride_elems] BF16
    const unsigned int rows,
    const unsigned int hidden,
    const unsigned int hc_mult,
    const unsigned int src_row0,
    const unsigned int dst_row_stride_elems,
    const unsigned int dst_slot_off_elems
) {
    const unsigned int r = blockIdx.x;
    if (r >= rows || hc_mult == 0u) return;

    const float* s = streams + (size_t)(src_row0 + r) * (size_t)hc_mult * (size_t)hidden;
    __nv_bfloat16* o = dst + (size_t)r * (size_t)dst_row_stride_elems
                           + (size_t)dst_slot_off_elems;
    const float inv = 1.0f / (float)hc_mult;

    for (unsigned int d = threadIdx.x; d < hidden; d += DSPARK_CAPTURE_BLOCK) {
        // FP32 accumulation over the streams, exactly as torch's `mean(dim=2)`
        // on the FP32 highway; only the store narrows to BF16.
        float acc = 0.0f;
        for (unsigned int i = 0; i < hc_mult; ++i) {
            acc += s[(size_t)i * (size_t)hidden + d];
        }
        o[d] = __float2bfloat16(acc * inv);
    }
}
