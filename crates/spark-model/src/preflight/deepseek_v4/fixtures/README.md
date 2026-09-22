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

# Pinned Vision-Exp tensor headers (native DSpark)

`vision-exp-tensor-headers.json` holds safetensors **headers only** (dtype,
shape, `data_offsets`) for both Vision-Exp packings, under two top-level keys:

| key | repo | revision |
|---|---|---|
| `official` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| `nvfp4` | `s-zaizen/DeepSeek-V4-Flash-Vision-Exp-NVFP4` | `325721df897916a21f71ba288b1c54f8de12d966` |

Regenerate with the script beside this directory (it embeds the revisions and
its own invocation in the fixture's `_generator` block):

```
/home/coder/.venvs/dsv4/bin/python3 \
  crates/spark-model/src/preflight/deepseek_v4/gen_tensor_header_fixture.py
```

Selection: every `mtp.*` tensor of the native 3-stage DSpark module, plus
target layer 42 for contrast, with routed experts sampled at 0 and 128 (EP rank
0 and rank 1 under EP=2). No tensor payload is downloaded — the index names the
shard and two HTTP byte-range reads give the header.

The load-bearing fact the tests pin: **the two packings differ only at the
TARGET layers.** `layers.42.ffn.experts.*` is MXFP4/E8M0 (`.scale`) in the
official packing and NVFP4 (`.weight_scale` + `.weight_scale_2` +
`.input_scale`) in the repack, while `mtp.{0,1,2}.ffn.experts.*` is MXFP4/E8M0
in BOTH. A DSpark loader that inferred the draft format from the target format
would be wrong on the repack.
