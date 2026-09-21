# DeepSeek V4 Flash 0731: local bring-up handoff

This is an **unvalidated implementation handoff**, not a support-matrix entry.
All compilation, tests and GPU validation below must run on the Sparks. No image
has been built, measured or published by this implementation session. Native
DSpark and complete sparse attention/request-state handling are separate work.

**Vision-Exp checkpoints are a different handoff.** This document covers the
text-only DeepSeek-V4-Flash-0731 checkpoints and pins branch
`feat/deepseek-v4-0731-bringup`. For `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
and `s-zaizen/DeepSeek-V4-Flash-Vision-Exp-NVFP4` — image input, ViT + aligner,
VL routing, the EP=2 aligner-row broadcast — use branch
`impl/deepseek-v4-flash-vision-exp` and read
[`DEEPSEEK_V4_VISION_EXP_BRINGUP.md`](DEEPSEEK_V4_VISION_EXP_BRINGUP.md), which
reuses the EP=2 procedure below and lists only the deltas.

The recipe is committed in `git@github.com:stalder-n/atlas-recipes.git`, on
branch `feat/local-recipe-resolution`, at
`examples/local/deepseek-v4-flash-0731-nvfp4-ep2.yaml`. Use that checked-in file;
no separately copied YAML is needed. It remains outside the published catalogue
and is selected by its file path.
It targets exactly two Sparks with TP=1, EP=2, one active request, eager execution,
FP8 KV and **2048 total tokens including output**. Keep speculation, prefix
caching and swapping disabled. Do not raise context or enable chunked prefill.
A successful startup does not establish numerical correctness.

## Check out both forks on each node

Atlas contains the `spark` inference server and kernels. The atlas-recipes
repository contains the `atlasctl` launcher as well as the recipe. Both branches
are required; building Atlas does not rebuild atlasctl.

For fresh checkouts on each Spark:

```bash
mkdir -p "$HOME/dev"
git clone --branch feat/deepseek-v4-0731-bringup \
  git@github.com:stalder-n/atlas.git "$HOME/dev/atlas"
git clone --branch feat/local-recipe-resolution \
  git@github.com:stalder-n/atlas-recipes.git "$HOME/dev/atlas-recipes"
```

For existing checkouts, fetch your forks, switch to these branches and pull with
`--ff-only` instead of cloning again. The recipes checkout's `origin` should be
`git@github.com:stalder-n/atlas-recipes.git`. Include Atlas fixes through
`c5d57d83` and recipes fixes through `574abe4` (or their descendants).

Set these paths in each node's shell, adjusting them to your checkout locations:

```bash
export ATLAS_SOURCE_DIR="$HOME/dev/atlas"
export ATLAS_RECIPES_DIR="$HOME/dev/atlas-recipes"
export ATLASCTL_BIN="$ATLAS_RECIPES_DIR/target/release/atlasctl"
export ATLAS_V4_RECIPE="$ATLAS_RECIPES_DIR/examples/local/deepseek-v4-flash-0731-nvfp4-ep2.yaml"
```

## Prepare pinned weights on both nodes

On each host, set `HF_HOME` to the directory containing the `hub/` directory
with your downloaded weights. atlasctl mounts this host directory at
`/cache/huggingface` inside the container; the identical-looking container path
does not automatically select `/cache/huggingface` on the host. For weights
stored on the host under `/cache/huggingface/hub`, use:

```bash
export HF_HOME=/cache/huggingface
test -r "$HF_HOME/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots/f1caa71142bd0be02f728c79f75042ac1e461579/config.json"
```

Keep this variable set in the shell that runs atlasctl. If unset, atlasctl uses
`$HOME/.cache/huggingface` as the host source. Inspect `run --print` and confirm
its volume maps the actual host cache to `/cache/huggingface`. Changing the
variable requires launching a new container; an existing container retains its
original mounts. A resolver error suggesting `hf download /cache/...` means the
snapshot was not visible as a local model inside that container; check mounts
before downloading again.

Using your existing Hugging Face client environment, on **each** Spark (skip
this download if the pinned snapshot is already complete):

```bash
hf download nvidia/DeepSeek-V4-Flash-0731-NVFP4 \
  --revision f1caa71142bd0be02f728c79f75042ac1e461579
```

atlasctl mounts its host Hugging Face cache at `/cache/huggingface` and sets
`HF_HOME` to that container path. Ensure both host caches contain the complete
snapshot and referenced blobs. The recipe's `model_from_path` is the exact
container-visible snapshot:

```text
/cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots/f1caa71142bd0be02f728c79f75042ac1e461579
```

`model_revision` alone does not enforce runtime pinning. Do not replace the
snapshot path with a mutable `refs/main` or a Hub model identifier.

## Build source and transfer the image (operator commands, not yet run)

From the Atlas branch above on a Spark. Implementation sources and fixtures are
committed; no untracked source archive is needed:

```bash
cd "$ATLAS_SOURCE_DIR"
cargo fmt --all -- --check
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
bash scripts/check-license-headers.sh
typos
python3 scripts/check_cross_hardware.py --base origin/main --worktree

# Compile and execute on the CUDA-capable Sparks, with actual kernels enabled.
ATLAS_TARGET_HW=gb10 cargo check --workspace
ATLAS_TARGET_HW=gb10 cargo test -p avarok-kernels --test deepseek_v4_kernel_audit -- --include-ignored
ATLAS_TARGET_HW=gb10 cargo test -p spark-model preflight::deepseek_v4
ATLAS_TARGET_HW=gb10 cargo test -p spark-server --bin spark v4_bringup
ATLAS_TARGET_HW=gb10 cargo test -p spark-server --bin spark context_budget
ATLAS_TARGET_HW=gb10 cargo test --workspace

docker build --no-cache -f docker/gb10/Dockerfile \
  -t atlas-deepseek-v4-0731:bringup .
docker image inspect atlas-deepseek-v4-0731:bringup \
  --format '{{.Id}} {{json .RepoDigests}}' > deepseek-v4-0731-image-identity.txt
git rev-parse HEAD > deepseek-v4-0731-source-sha.txt
git diff --binary > deepseek-v4-0731-source.patch
docker save -o deepseek-v4-0731-image.tar atlas-deepseek-v4-0731:bringup
```

Transfer the image tar and identity record to the other Spark using your
normal file transfer. The recipe comes from its checkout on each node. On the
second Spark:

```bash
docker load -i deepseek-v4-0731-image.tar
docker image inspect atlas-deepseek-v4-0731:bringup \
  --format '{{.Id}} {{json .RepoDigests}}'
```

Check identical image IDs on both nodes. A locally built image may have no
registry `RepoDigests`; record its actual immutable image ID and leave registry
digest unset. No digest is asserted here before a successful build. Publish
nothing. Use the general GB10 Dockerfile because it includes the benchmark
fixtures required by the source build.

Build the modified atlasctl checkout on each node:

```bash
cd "$ATLAS_RECIPES_DIR"
cargo fmt --all -- --check
cargo test -p atlasctl-core registry
cargo test -p atlasctl-core docker::translate
cargo clippy --workspace --tests
cargo build --release -p atlasctl
git rev-parse HEAD > /tmp/deepseek-v4-0731-atlas-recipes-sha.txt
```

## Inspect and render both ranks

Set `ATLAS_MASTER_ADDR` to rank 0's reachable RoCE-network address on both
machines; use the same address and port on both. Run rank 0 only on the first
Spark and rank 1 only on the second. `--print` renders without launching.
Plain `recipe show` displays local-path provenance; for this two-node
recipe use `run --print` to render a rank (the current `show --docker` interface
has no placement arguments).

```bash
# On either node:
"$ATLASCTL_BIN" recipe show \
  "$ATLAS_V4_RECIPE"

# On rank 0 only:
"$ATLASCTL_BIN" run \
  "$ATLAS_V4_RECIPE" \
  --rank 0 --world-size 2 --master-addr "${ATLAS_MASTER_ADDR:?set rank 0 address}" \
  --master-port 29500 --no-pull --print

# On rank 1 only:
"$ATLASCTL_BIN" run \
  "$ATLAS_V4_RECIPE" \
  --rank 1 --world-size 2 --master-addr "${ATLAS_MASTER_ADDR:?set rank 0 address}" \
  --master-port 29500 --no-pull --print
```

Verify both commands use the exact snapshot path, TP1/EP2, FP8 KV, context 2048,
batch 1, max-num-seqs 1, max-prefill-tokens 2048, swap-space-gb 0,
utilization 0.90, guard 512 MB and `ATLAS_EP_GRAPHS=0`. Rank 0 must
serve port 8888 and rank 1 port 0. Boolean false settings should omit their
opt-in flags. Launch the matching command on each node with `--print` removed.
The runtime capability checks remain authoritative if CLI overrides are used.

## Required evidence before calling this runnable

Retain both startup logs, actual image identity, source revision plus uncommitted
patch (if any), atlas-recipes revision, checked-in recipe, rendered commands and
checkpoint revision. Run loader format
positive/negative tests and old-checkpoint regression tests, then short numerical
reference comparisons. Verify 128 routed experts belong to each EP rank.

Send two sequential requests and compare the second result with a fresh server
using deterministic decoding, to expose compression-state leakage. Exercise
chat, low/high/max reasoning, tool results, and streamed tools through OpenAI
and Anthropic endpoints. Check prompt plus requested output above 2048 is
rejected. Run the relevant `tests/single_gpu_suite.py` cases and model regressions
on the appropriate hardware; use the suite's current `--help` for case selection.

Every command and numerical/serving check in this document is **unrun** in this
container. Performance certification requires the measurement and certification
oracles first. No throughput, maximum usable context or DSpark speedup is claimed.

The model registry metadata now names the 0731 reference checkpoint and its
checkpoint positional limit; that field is not a qualified serving context.
The server's 2048-token bring-up limit remains authoritative. The kernel reach
check above is required because `MODEL.toml` changed; run the cross-hardware
oracle before pushing. No CUDA kernel implementation changed.

Also run the draft upload-policy regression:

```bash
ATLAS_TARGET_HW=gb10 cargo test -p spark-server --bin spark draft_loading_tests
```

Source inspection confirms that batch size limits active plus prefilling
sequences, and whole-prompt V4 prefill resets compression counters and seeds
ring state. This is not GPU evidence of isolation: test sequential requests,
a one-token completion prompt, cancellation and requeue before serving traffic.


## First EP=2 startup audit failure

The supplied rank logs both stopped on two unresolved optional probes:
`inferspark_prefill_512tc::inferspark_prefill_512tc` and
`moe_silu_mul::silu_mul_quant_fp8`. Commit `c5d57d83` corrects the first
probe's module name in the runtime exemption and adds the second probe's
runtime declaration alongside its existing static shadow exemption. The latter
preserves the unfused V4 SwiGLU clamp; the common fused kernel is not a valid
replacement. The allocation sweep warnings followed the abandoned load.

Pull this fix, run the audit regression above on a Spark, rebuild the image,
transfer it to the other node and recreate both rank containers. The target
metadata is embedded at build time, so updating the recipe or restarting old
containers does not install the fix. This has static verification only in this
session; the new image's startup and numerical behavior still require validation.


## Expanded smoke validation (standalone script version 2)

The local helper is `/tmp/atlas-v4-smoke.py`; copy it to rank 0 as
`atlas-test.py`. It requires only Python's standard library. Both rank
containers must be running, with no other clients sending inference requests.
The operator's current API port is 37777; adjust the URL to the actual listener.

```bash
python3 atlas-test.py http://127.0.0.1:37777 v4-expanded-first
# After the first run passes, restart BOTH rank containers, wait until ready:
python3 atlas-test.py http://127.0.0.1:37777 v4-expanded-fresh \
  --compare v4-expanded-first
```

Version 2 counts prompts with `/tokenize`, verifies those counts against actual
completion usage, and exercises 127/128/129, 255/256/257, 1024 and 1536 prompt
tokens. It checks retrieval, complete counting from 1 through 128, sequential
repeatability, blocking/streaming agreement, recovery after a client closes a
partial stream, and the context guard. The disconnect probe observes a client
socket close before a terminal event; it does not prove internal server cleanup.

Each output directory must be new. Requests, responses, fixtures, cases and a
summary are saved there; the script stops at the first failure. `--compare`
requires a passing version-2 run, not the original short smoke-test results.
Strict answer/format failures require investigation and are not automatically
proof of a kernel defect. The suite does not certify reference numerics, tool
calling, thinking, or context beyond the bring-up limit. This session checked
the script syntax only; execute it on the Sparks.
