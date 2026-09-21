# Pinned NVIDIA 0731 checkpoint fixtures

Source: `nvidia/DeepSeek-V4-Flash-0731-NVFP4`, revision
`f1caa71142bd0be02f728c79f75042ac1e461579`.

`0731-config.json` is the unmodified config fetched from the pinned Hugging
Face resolve URL. `0731-tensor-headers.json` contains unmodified entries selected
from safetensors headers using HTTP byte-range requests and that revision's
`model.safetensors.index.json`. No tensor payload is included. Offsets refer to
the original shards, not to a combined file.

The sample covers routed experts 0 and 128, all three shared projections,
attention `wq_a`, and stage-zero draft expert 0. Base experts use U8, E4M3
block scales (group 16), and zero-dimensional F32 global/input scales. Draft
experts use packed I8 and E8M0 block scales (group 32); the runtime maps I8
packed containers to UInt8 without numerical conversion.

Tests use null device pointers and validate headers only. Synthetic negatives
are derived in test code and are not represented as upstream data.
