# DeepSeek V4 opt-in prefill and DSML validation

These additions do not expand the bring-up envelope: TP1/EP2, eager execution,
one active request, at most 2,048 prompt plus output tokens. Keep speculation,
prefix caching, swapping and chunked prefill disabled. Use the pinned
`nvidia/DeepSeek-V4-Flash-0731-NVFP4` snapshot
`f1caa71142bd0be02f728c79f75042ac1e461579` on both Sparks.

The two numerical changes are independently opt-in and resolve through
`ModelLevers` when loading. Neither changes weight residency or cache behavior.
DSML grammar uses the existing tool policy, grammar cache and API routing.

| Review unit | Main paths | Behavior |
| --- | --- | --- |
| Native FP8 prefill | `spark-model/.../prefill/v4_projections.rs`, projection call sites, ModelLevers | W8A16 pipelined GEMM for contiguous projections; grouped strided GEMV batches of at most 16 tokens for `wo_a`; BF16 fallbacks selected before launch |
| Compressed attention | V4 `prefill_attn_compressed_tc.cu`, kernel registration, optional handle and attention dispatch | BF16 WMMA QK tiles of 32 queries by 16 keys; FP32 softmax and value accumulation, shared raw/compressed denominator and sink |
| DSML | `spark-server/src/grammar/deepseek_v4.rs`, parser and tokenizer wiring, `xgrammar/src/schema/gen_dsml.rs` and UTF-8 tag decoding | Schema-constrained registered tools, typed parameters, canonical DSML markers, multiple invocations |

**Shared converter, UTF-8 tag decoding and grammar-routing changes need explicit reviewer attention.**
The tag decoder must reconstruct UTF-8 strings from stored bytes so canonical
DSML triggers and raw-string closing delimiters reach the matcher intact.
Run existing DeepSeek XML, other tool-format, grammar-policy and both API-surface
regressions together with the new DSML cases. A passing parser test alone does
not establish that the matcher accepts valid generation or masks invalid tokens.

## Host checks

```bash
cargo fmt --all -- --check
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --workspace
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo doc --workspace --no-deps
bash scripts/check-license-headers.sh
typos
python3 scripts/check_kernel_shadows.py
python3 scripts/check_cross_hardware.py --base origin/main --worktree
python3 -m unittest discover -s tests -p test_deepseek_v4_prefill_compare.py
```

On a CPU-only runner, linking server tests also requires the fail-fast GPU stubs
from `scripts/ci_gpu_stubs.sh`, as in CI. They are linker support only: GPU tests
must run against real CUDA libraries on a Spark. Do not count skipped GPU tests
as passes. The cross-hardware checker does not include untracked kernel files;
include the new file explicitly in an attribution check before staging:

```bash
python3 scripts/check_cross_hardware.py --paths \
  kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed_tc.cu \
  kernels/gb10/deepseek-v4-flash/nvfp4/KERNEL.toml
```

## Spark kernel checks and clean image

Use a CUDA-enabled Python environment with PyTorch, pytest, ninja and the CUDA
compiler for the numerical harnesses. They compile the production CUDA sources
and compare against separately dequantized or float64 reference calculations.
Run with the compiler and GPU architecture used for the image.

```bash
python3 -m pytest -q tests/gpu/test_v4_fp8_prefill.py
python3 -m pytest -q tests/gpu/test_v4_compressed_attention_tc.py
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4-flash \
  ATLAS_TARGET_QUANT=nvfp4 cargo test -p avarok-kernels \
  --test deepseek_v4_kernel_audit -- --include-ignored
```

The FP8 harness covers each projection, independent native-weight dequantization,
nontrivial scales, grouped strides and token boundaries. Attention covers ratios
4/128, partial tiles, sliding windows, empty compressed history, sink extremes,
causality perturbations, finite outputs and output-buffer sentinels. Also run
under `compute-sanitizer --tool memcheck` on the Spark to detect invalid reads.

After committing the reviewed source, build once from scratch and transfer that
same image to the other Spark. Do not publish or move release tags at this stage.
Use the existing local EP2 recipe and its `model_from_path` pinned snapshot.

```bash
export V4_SOURCE_SHA=$(git rev-parse HEAD)
docker build --no-cache -f docker/gb10/Dockerfile \
  --build-arg ATLAS_GIT_SHA="$V4_SOURCE_SHA" \
  -t "atlas-v4-validation:$V4_SOURCE_SHA" .
docker image inspect "atlas-v4-validation:$V4_SOURCE_SHA" \
  --format '{{.Id}} {{json .RepoDigests}}'
docker save -o /tmp/atlas-v4-validation.tar "atlas-v4-validation:$V4_SOURCE_SHA"
# Transfer the tar to rank 1, then there:
docker load -i /tmp/atlas-v4-validation.tar
```

Record identical image IDs on both nodes; a local image may have no registry
digest. Retain source SHA, any source patch, recipe revision, both rendered Docker
commands, both startup logs and `docker inspect` output. The image must include
the V4 target; a successful host build with `ATLAS_SKIP_BUILD=1` is insufficient.

## Controlled whole-prefill logits

Run the four configurations sequentially, recreating **both** rank containers
between modes. Put these environment variables explicitly in both containers
using the existing recipe `env` map or `docker run -e` options; shell exports
alone do not establish that atlasctl forwards them.

| Mode | `ATLAS_V4_FP8_PREFILL` | `ATLAS_V4_COMPRESSED_ATTN_TC` |
| --- | --- | --- |
| baseline | `0` | `0` |
| fp8 | `1` | `0` |
| tc | `0` | `1` |
| both | `1` | `1` |

On rank 0, add `-v /tmp/v4-dump:/v4-dump -e ATLAS_NEMO_DUMP=/v4-dump`
to the rendered Docker command before the image argument, after creating the
host directory. This existing diagnostic writes the final whole-prefill raw
logits as little-endian FP32 to `atlas_logits.bin`. Keep it set identically in
all four modes. It also writes per-layer diagnostics, so these runs are for
correctness and must not be used as performance measurements.

Create `fingerprint.json` with these required keys, filling values from actual
records: `source_sha`, `image_id`, `checkpoint`, `checkpoint_revision`,
`rank0_host`, `rank1_host`, `rank0_serve_args`, `rank1_serve_args`,
`common_environment`, `recipe_sha`, `vocab_size`. Use the checkpoint config for
vocabulary size. Record all serving flags and environment variables except the
two toggles in this common fingerprint; keep the full per-mode Docker records
alongside it. This is operator-attested metadata: the API cannot verify it.

For each mode, after both ranks are ready and without other clients:

```bash
python3 tests/deepseek_v4_prefill_compare.py collect \
  --base-url http://127.0.0.1:8888 \
  --model nvidia/DeepSeek-V4-Flash-0731-NVFP4 \
  --fingerprint fingerprint.json --dump-dir /tmp/v4-dump \
  --mode baseline --output v4-baseline
# Repeat after restarting both ranks with fp8, tc, both; change mode/output.
```

The harness sends exact token-ID prompts spanning batch/tile/window boundaries,
requests one output token, checks reported prompt lengths, saves raw responses
and captures a fresh full logit vector per prompt. Its longest prompt is 2,047
tokens. It deletes only the previous `atlas_logits.bin` before each request to
reject missing/stale diagnostic output. Use an exclusive diagnostic directory.

Choose and record whole-model tolerances **before** inspecting candidates; the
projection microtest tolerance is not automatically a whole-model error bound.
The comparator requires explicit absolute and relative tolerances and reports
argmax changes separately. Add `--require-argmax` to enforce exact raw argmax.

```bash
python3 tests/deepseek_v4_prefill_compare.py compare \
  --reference v4-baseline --candidate v4-fp8 --output v4-fp8-comparison.json \
  --atol "${V4_LOGIT_ATOL:?set predeclared absolute tolerance}" \
  --rtol "${V4_LOGIT_RTOL:?set predeclared relative tolerance}"
# Repeat against v4-tc and v4-both with identical thresholds.
```

Any nonfinite vector, absent row, changed fingerprint or request, or element
outside `atol + rtol * abs(reference)` fails the comparison. Capture/compare a
fresh baseline again to separate repeatability from a toggle effect. A finite
output or matching answer alone does not prove numerical parity.

## Serving and certification stages

Exercise sequential requests, streaming DSML split delimiters, reasoning-to-tool
transitions and auto/required/named/none tool policy on both OpenAI and Anthropic
endpoints. Keep the native prompt encoding. Validate typed/nested arguments,
multiple invocations, unknown names, malformed envelopes and absent required
parameters with actual matcher/token-mask tests as well as end-to-end requests.
Use the relevant bounded cases from `tests/single_gpu_suite.py`; its full suite
includes cases outside this V4 bring-up envelope, so inspect case budgets first.
Run other supported-model regressions before calling the image verified.

Performance remains unmeasured until controlled runs on the Sparks. Follow
`.claude/skills/measurement-discipline/SKILL.md`: identical fingerprints except
for the toggle, evidence that the intended kernel engaged, diagnostics disabled,
and at least three repeats for any headline claim. Never infer speedup from
kernel selection or reduced launch counts.

Before pushing kernel changes run the cross-hardware oracle. Only after code
and image are frozen, initiate certification through
`.claude/skills/oracle_certification_state_check/SKILL.md` (`pre`, `begin`,
`during`, `post`, `release`). Do not start a campaign on a dirty perf tree or
bank records from a different source/image. Run the release skill's applicable
serve-matrix/image gates before any shippable tag. Both numerical toggles stay
off by default regardless of CPU CI results.

## Implementation-session verification

On the CPU development host, the final combined tree passed workspace tests,
clippy, formatting, rustdoc, typos, kernel shadow structure and cross-hardware
reach checks. Workspace tests used the repository's fail-fast GPU linker stubs
under `/tmp` and unset the session's `NO_COLOR=1` for the existing TUI color
assertions. The raw-logit comparator's five host tests passed. New GPU harnesses
passed Python syntax checks only.

The final DSML focused suite passed 11 tests, including production prewarm and
split-token masking with a required string parameter. The shared xgrammar unit
suite passed 838 tests. A UTF-8 regression verifies trigger, stop and exclusion
markers, with a separate check preserving legacy invalid-byte handling.

No CUDA compiler, GPU or Docker was available on this host. Consequently the
GPU numerical suites, clean image build/serve validation and certification were
not run. The Docker-based license engine also could not run; a direct inspection
confirmed SPDX headers on changed source files, without claiming equivalence to
that engine. Both numerical switches remain default-off. No performance result
or correctness qualification on the pinned checkpoint is asserted here.
