# DeepSeek-V4-Flash-Vision-Exp: Spark bring-up handoff

This is an **unvalidated implementation handoff**, not a support-matrix entry.
Everything below was written on a host with **no GPU and no `nvcc`**.

Read this together with
[`DEEPSEEK_V4_0731_LOCAL_BRINGUP.md`](DEEPSEEK_V4_0731_LOCAL_BRINGUP.md): the
two-node EP=2 procedure (checkouts, `HF_HOME` mounts, image build and transfer,
`run --print` rendering, rank placement, required evidence) is *that* document's,
and is not repeated here. This one covers only what the Vision-Exp checkpoints
change. Branch: `impl/deepseek-v4-flash-vision-exp`.

**No CUDA kernel on this branch has ever been compiled.** The five kernels added
for this work — `vision_encoder` (a same-hardware symlink of the Qwen ViT
source), `vision_deepseek` (3×3 pixel unshuffle, exact-erf GELU, unclamped
SwiGLU split), `prefill_attn_compressed_vis` (image-span key bounds),
`moe_router_vl` (per-modality routing bias) and `dspark_capture` (the mHC-stream
mean the DSpark drafter is fed) under
`kernels/gb10/deepseek-v4-flash/nvfp4/`, plus their hopper/b200 mirror symlinks —
are blind code. The tests that would catch a wrong launch signature read the PTX
baked into the binary, so they are `#[ignore]` **and vacuous on a stub build**:

- `crates/avarok-kernels/tests/kernel_arity.rs::w4a16_launch_family_arity_pins`
- `crates/avarok-kernels/tests/deepseek_v4_kernel_audit.rs::compiled_v4_registry_carries_the_optional_probes`
- `crates/avarok-kernels/tests/deepseek_v4_kernel_audit.rs::compiled_v4_registry_carries_the_vision_modules`

They only mean anything on the Spark, run with `-- --include-ignored` and with
`AVAROK_SKIP_BUILD` **unset**. Run them before anything else.

---

## 1. Checkpoints

| | official | s-zaizen repack |
|---|---|---|
| HF id | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `s-zaizen/DeepSeek-V4-Flash-Vision-Exp-NVFP4` |
| Revision (pin this) | `6821d6ad3681a4b137b066b76094fa82ebd0a380` | `325721df897916a21f71ba288b1c54f8de12d966` |
| License | MIT | MIT |
| Shards / size | 48 · ≈168 GB | 48 · ≈176 GB |
| Attention / shared / MTP | FP8 block-scaled (`weight_block_size [128,128]`, `scale_fmt ue8m0`) | same |
| Routed experts | `expert_dtype: "fp4"` → Atlas MXFP4-E8M0 | NVFP4, `group_size 16` |

Both declare `model_type: deepseek_v4`, `hidden_size 4096`, `vocab_size 129280`,
`rms_norm_eps 1e-20`, `sliding_window 128`, `num_nextn_predict_layers 3`, and an
identical flat vision block: `vision_n_layers 32`, `vision_dim 1024`,
`vision_n_heads 16`, `vision_inter_dim 2816`, `vision_patch_size 14`,
`vision_rope_theta 10000.0`, `vision_downsample_ratio 3`, `vision_max_n_token
384`, `vision_min_pixels 147456`, `vision_max_wh_ratio 8`.

**Read the repack's quantization block carefully.** Its `config.json` still says
`quantization_config.quant_method: "fp8"`; the NVFP4 part is in the *same* block
as `quant_algo: "MIXED_PRECISION"`, `moe_quant_algo: "NVFP4"`, `group_size: 16`
and a per-layer `quantized_layers` map (`layers.N.ffn.experts` → NVFP4), mirrored
in a separate `hf_quant_config.json`. An operator who reads only `quant_method`
will conclude this is a plain FP8 checkpoint. It is not.

Useful for validation: **all 267 `vision.*` / `aligner.*` /
`image_{start,pad,newline,end}` tensors live in
`model-00001-of-00048.safetensors` (≈2 GB) in *both* packings.** The reference
compare script opens that one shard, not 168 GB.

Both nodes need the complete snapshot — EP=2 splits the experts at load time, not
on disk. Pin by snapshot path exactly as the 0731 document does; `model_revision`
alone does not pin at runtime.

---

## 2. Build and check on the Spark

The kernels were never compiled, so this is step one, before any serving:

```bash
AVAROK_TARGET_HW=gb10 cargo check --workspace
AVAROK_TARGET_HW=gb10 cargo test -p avarok-kernels -- --include-ignored
AVAROK_TARGET_HW=gb10 cargo test -p spark-model vision_deepseek
AVAROK_TARGET_HW=gb10 cargo test -p spark-server deepseek_v4
AVAROK_TARGET_HW=gb10 cargo test --workspace -- --include-ignored
```

> **`AVAROK_TARGET_HW`, not `ATLAS_TARGET_HW`.** `crates/avarok-kernels/build.rs`
> reads only the `AVAROK_*` names; `avarok_core::env_compat::mirror_legacy_env`
> runs in the server binary's `main`, not in a cargo build script, and
> `ATLAS_SKIP_BUILD` is the one legacy name with an explicit build.rs fallback.
> The 0731 document's `ATLAS_TARGET_HW=gb10` lines appear to work only because
> `build_diagnose::DEFAULT_HW` is already `gb10`.

Host gates (they pass on the implementation host; re-run them after any fix):

```bash
cargo fmt --all -- --check
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
bash .github/scripts/check-file-size-cap.sh
typos
```

---

## 3. EP=2 recipe — deltas against the 0731 document

Follow [`DEEPSEEK_V4_0731_LOCAL_BRINGUP.md`](DEEPSEEK_V4_0731_LOCAL_BRINGUP.md)
for the whole two-node procedure. Change only these things:

- **`model_from_path`** → the Vision-Exp snapshot for the packing you chose
  (§1), on **both** nodes.
- **The bring-up envelope is enforced in code, not by convention.**
  `crates/spark-server/src/main_modules/serve_phases/v4_bringup.rs` refuses to
  start otherwise: speculation off (`--speculative`, `--self-speculative`,
  `--ngram-speculative`, `--dflash` are all rejected), `--max-batch-size 1`,
  `--max-seq-len` in `1..=2048` (**prompt plus output**), prefix caching off,
  `--swap-space-gb 0` with high-speed swap off, and
  `--max-prefill-tokens >= --max-seq-len` (whole-prompt prefill; chunk
  continuation would reset the compressor at the wrong offset).
- **Budget arithmetic for images.** `vision_min_pixels` (147456) floors *every*
  image at a 10×10 aligner grid = **114 sentinel tokens** plus 0–3
  compression-pad tokens, so **117 tokens is the cheapest possible image**. The
  `vision_max_n_token 384` cap makes the largest block **≈369**. Worked examples
  from the committed fixtures, computed with the reference solver:

  | image | best size | ViT grid | LLM grid | tokens (at `start_pos % 4 == 0`) |
  |---|---|---|---|---|
  | `tests/fixtures/mona_lisa.jpeg` 1024×1024 | 756×756 | 54×54 | 18×18 | 349 |
  | `images/05_square_768.png` 768×768 | 756×756 | 54×54 | 18×18 | 349 |
  | `images/03_landscape_512x384.png` | 518×392 | 28×37 | 10×13 | 145 |
  | `images/06_wide_1024x576.png` | 1036×588 | 42×74 | 14×25 | 369 |
  | `images/01_square_224.png` | 392×392 | 28×28 | 10×10 | 117 |

  With the 2048-token ceiling covering prompt **and** output, that is roughly
  five images per request at best, one or two in practice.
- **Do not set `--vision-max-pixels`.** It is a Qwen pixel-area bound and is
  meaningless for the DeepSeek tower, whose sizing comes from
  `vision_max_n_token`. The server is required to reject it when the config
  resolves to `VisionFlavor::DeepseekV4`; if a build silently accepts it, that is
  a defect — leave the flag unset either way.
- **Rank 0 owns the ViT; rank 1 never loads it.** Expect rank 1's startup log to
  show no vision-tower load and roughly 0.9 GB less resident memory. Rank 0
  broadcasts the aligner rows to rank 1 right after the token broadcast.
- **Optional, rank 0 only** — the reference-compare dump:
  `-e AVAROK_DS_VISION_DUMP=/tmp/ds-vision-dump -v /tmp/ds-vision-dump:/tmp/ds-vision-dump`.
  The encoder is required to write, per image `i`, `image_<i>.bf16` (raw BF16,
  row-major `[n_llm_h · n_llm_w, hidden]`, **unpermuted** aligner order) and
  `image_<i>.json` (`n_vit_h`, `n_vit_w`, `n_llm_h`, `n_llm_w`). An empty
  directory after a request means the variable is not set on rank 0, or the
  mount does not reach the host.
- `AVAROK_EP_GRAPHS=0`, as in the 0731 recipe.

---

## 4. Numeric validation ladder

Run these in order. Each rung is only meaningful if the one above it passed.

**1 — host reference against the real weights.** Loads `vision.*` / `aligner.*`
out of the checkpoint into the f32 host reference and prints the `encode_image`
checksum:

```bash
DSV4_VISION_CKPT=<snapshot> \
  cargo test -p spark-model --lib vision_deepseek_ref -- --ignored --nocapture
```

**2 — Atlas against DeepSeek's own reference code.** Start rank 0 with
`AVAROK_DS_VISION_DUMP`, then:

```bash
python3 tests/deepseek_v4_vision_compare.py --mode all \
  --base-url http://127.0.0.1:8888/v1 --model <served-id> \
  --checkpoint <snapshot> --dump /tmp/ds-vision-dump \
  --image tests/fixtures/images/05_square_768.png \
  --output /tmp/ds-vision-compare.json
```

The script fetches `inference/vision.py` and `inference/image_processor.py` at
revision `86f746b36186f0e567729a5c06a8c918caba82a9`, **verifies a committed
sha256 per file before importing them** (this code is executed; there is no
bypass flag), runs the tower on CPU in float32, and compares the dumped aligner
rows. Geometry is checked first — a grid mismatch is reported before any numeric
work, because the numbers would be meaningless. Its `--min-cos` (default 0.99) is
a bring-up signal, **not a certified tolerance**.

Its own discrimination can be checked anywhere, with no server and no
checkpoint (needs torch, Pillow, numpy, safetensors):

```bash
python3 tests/deepseek_v4_vision_compare.py --mode selftest
```

**3 — isolate the tower from the resampler.** Repeat rung 2 with a 504×504 PNG.
Verified against the reference solver: 504×504 resolves to `best 504×504`, ViT
36×36, LLM 12×12, 158 tokens — *no resize and a no-op pad*. Any residual
difference there is the tower, not Pillow parity. Use PNG, so JPEG decoding
cannot enter either:

```bash
python3 -c "from PIL import Image; \
  Image.open('tests/fixtures/mona_lisa.jpeg').convert('RGB') \
       .resize((504,504)).save('/tmp/ds-504.png')"
```

**4 — serving smoke.**

```bash
python3 tests/single_gpu_suite.py --base-url http://127.0.0.1:8888/v1 \
  --model <served-id> --vision --vision-ds --skip-longctx
```

`--vision-ds` adds the DeepSeek-specific cases: two images in one turn (two
different grids and two different `start_pos % 4` phases), an image inside a
`role:"tool"` array result, and repeat determinism. Each pairs a structural
check with a `usage.prompt_tokens` delta against a text-only twin request, so an
image that was silently dropped fails instead of producing a plausible answer.
It is opt-in and must **not** be sent to other vision models.

**5 — sequential-request isolation.** The suite's repeat-determinism case, then
the 0731 document's standalone smoke (version 2) with `--compare` after
restarting **both** rank containers. This is the compressor-state-leak probe.

**6 — budget rejection.** A prompt plus requested output above 2048 must return
**400**, not 500 and not a hang. An image prompt that cannot be prefilled whole
must also return 400 — chunked prefill for image prompts is refused by design,
not supported.

**7 — tools and thinking on both surfaces.** Protocol drift between the OpenAI
and Anthropic paths is this codebase's named historical failure mode, so exercise
both: OpenAI `/v1/chat/completions` (single image, multi-image, an image inside a
`role:"tool"` array content, a streamed tool call) and Anthropic `/v1/messages`
(an `image` block, a `tool_result` carrying an image, thinking).

---

## 5. Accepted deviations from "exact reference parity"

- **EXIF orientation.** The reference does a bare `Image.open(...).convert("RGB")`
  and Pillow does **not** honour the EXIF orientation tag. Atlas applies it, as
  the Qwen path does, so a phone photo stored with `Orientation = 6` is not
  handed to the model a quarter-turn over. Deliberate; documented at
  `crates/spark-model/src/vision_deepseek/preprocess.rs`.
- **Resampling.** Pillow's bicubic (`Resample.c`) and `ImageOps.pad` are ported
  to Rust and pinned by goldens recorded from Pillow itself
  (`vision_deepseek/fixtures/resample.json`), including the u8-quantised
  intermediate, the downscale support widening and Python's banker's rounding.
  Parity is proven for those cases, not for arbitrary images.
- **Image decoding.** Atlas decodes with the Rust `image` crate, the reference
  with Pillow/libjpeg. JPEG output can differ by a least significant bit before
  any of our code runs — prefer PNG inputs when comparing numbers.
- **Precision.** The reference oracle runs f32 on CPU; Atlas runs BF16 with f32
  accumulation. Expect small, uniform differences, not zeros.
- **DSpark (Phase 3) is greedy.** The native 3-stage chain accepts a greedy draft
  chain; vLLM's probabilistic draft sampling is not implemented.
- **No chunked prefill** for image prompts (400), **no prefix caching** for
  prompts containing sentinels, **no video**, **no `--vision-max-pixels`**.

---

## 6. Explicitly unvalidated

- No CUDA kernel on this branch has been compiled, anywhere, ever.
- No numeric, throughput, usable-context or image-quality claim is made.
- The EP=2 aligner-row broadcast has never been exercised on hardware.
- The **official FP8/fp4 packing has never been loaded by Atlas** — only
  NVIDIA's text-only 0731 NVFP4 checkpoint was ever attempted, and that attempt
  is itself unvalidated (see the 0731 document's startup-audit section).
- `vision = true` in `kernels/gb10/deepseek-v4-flash/MODEL.toml` has **no Rust
  reader**; it is documentation. Vision presence is decided at runtime from
  `config.json` (`vision_n_layers > 0`).
- `tests/deepseek_v4_vision_compare.py` has been exercised only against a
  synthetic checkpoint and a synthetic dump (`--mode selftest`). It has never
  seen a dump written by Atlas.
- The suite's prompt-token assertions assume `usage.prompt_tokens` counts the
  **expanded** sentinel stream. If the server reports a pre-expansion count, both
  new cases fail spuriously; the failure text says so.

---

## 7. Native DSpark (Phase 3)

**Implemented, never executed.** Both startup refusals are lifted — the
`!use_speculative` bail in `crates/spark-model/src/preflight/deepseek_v4.rs`
and the `args.speculative` clause of `validate_v4_bringup` in
`crates/spark-server/src/main_modules/serve_phases/v4_bringup.rs`, the latter
only for a checkpoint that declares `dspark_block_size > 0`. Every other
speculation mode (`--self-speculative`, `--ngram-speculative`, `--dflash`)
stays refused on V4.

### 7.1 What runs

```
target step   layers 40/41/42 → mean over the mHC streams → capture band
                                (dspark_capture.cu :: dspark_hc_mean_capture)
propose()     ring catch-up : main_x = main_norm(main_proj(capture))
                              ring[s][pos % 128] = rope(kv_norm(wkv_s(main_x)))
              block         : x = hc_expand(embed([last_token, noise × 4]))
              stage s=0,1,2 : hc_pre → attn_norm → MLA(q@pos+1..pos+5,
                                K = [valid ring ‖ block KV], bidirectional)
                              → hc_post → hc_pre → ffn_norm → MoE → hc_post
              head          : x = hc_head(h); logits = lm_head(norm(x))
                              greedy Markov chain → 5 drafts
                              confidence = conf_w · [x_i ; markov_embed_i]
```

Files: `crates/spark-model/src/layers/deepseek_v4_dspark{.rs,/}`,
`crates/spark-model/src/weight_loader/deepseek_v4/dspark.rs`,
`crates/spark-model/src/model/impl_b3_dspark.rs`,
`kernels/gb10/deepseek-v4-flash/nvfp4/dspark_capture.cu` (+ hopper/b200
symlinks).

### 7.2 Served at K=4, not K=6

The checkpoint's trained block size is **5**; Atlas serves **`num_drafts = 3`**
(verify K=4) and `serve_load.rs` clamps anything higher with a warning.
`scheduler/mtp_step.rs` routes `drafts.len() >= 4` to `step_verify_dflash`,
whose model hooks (`dflash_accept_append`, `dflash_eagle_accept_append`,
`dflash_eagle_kgamma_append`) hard-error on any state that is not a
`DflashProposerState`. 3 is also vLLM's measured optimum for this checkpoint,
so the clamp costs nothing today; raising it means teaching that verify ladder
about DSpark, which is a separate unit.

### 7.3 Acceptance expectation, and what it is NOT

vLLM measures **2.99 accepted tokens per forward at K=3** on this checkpoint.
That is the number to compare against — and it is *necessary, not sufficient*:

* `MoeLayer::forward_prefill` at 5 rows falls below the grouped-GEMM threshold
  (64) into `forward_batched`, which re-streams the expert weights per token.
  **Three stages × 5 rows × 256 experts of routing per propose may make DSpark
  net-negative on throughput even at the expected acceptance.** Measure wall
  time, not acceptance, before believing anything — and measure it through
  `/measurement-discipline` and the certification oracle, never by eye.
* Loading three full V4 stages adds roughly **10 GB replicated / 5 GB per rank
  under EP=2** (the stages are EP-sharded, `force_all_experts = false`, and are
  loaded on BOTH ranks because the propose all-reduces). If the profile is
  already near the memory ceiling, `--speculative` may simply not fit.

### 7.4 Deviations from the reference, all deliberate

| # | Reference | Atlas | Why |
|---|---|---|---|
| 1 | `sample(logits, temperature)` per chain row | greedy `argmax` | probabilistic draft sampling is out of scope; a greedy chain only costs acceptance, never correctness — every draft is re-verified |
| 2 | `act_quant(kv[..., :-rd], 64, …, inplace=True)` — the KV latent round-trips through FP8 | BF16 ring | Atlas's ring is BF16; the golden generator monkeypatches `act_quant` to identity and records it |
| 3 | keys `[ring[0..min(win,pos+1)) ‖ ring[win .. win+γ)]` | block KV written at index `ring_len` so the key range is ONE interval | `prefill_attn_compressed_vis` takes one inclusive `[lo,hi]` per query; softmax over keys is permutation-invariant and RoPE is applied at write time |
| 4 | confidence in f32 end to end | the two half-dots are BF16 GEMVs, summed in f32 on the host | only affects the `AVAROK_MTP_DRAFT_CONF` gate, which is staged off |

### 7.5 What is verified here, and what the Spark must verify

Host-verified (`cargo test -p spark-model deepseek_v4_dspark`,
`preflight::deepseek_v4`):

* stage discovery and the quant contract on BOTH packings, from
  `preflight/deepseek_v4/fixtures/vision-exp-tensor-headers.json`;
* ring index algebra — slot, length, the wrap, the prefill `cutoff` split, the
  block-KV placement, catch-up row selection and gap reporting;
* capture-band addressing;
* **the greedy Markov chain and the confidence dot against DeepSeek's own
  `forward_spec` outputs** (`layers/deepseek_v4_dspark/fixtures/dspark_tiny_golden.json`,
  generated by `gen_dspark_tiny_golden.py` from the reference at the pinned
  revision), fed the golden's own `x` and pre-Markov logits so the stage bodies
  are out of the picture;
* the inert-when-disabled guard.

NOT verified anywhere on this host, in descending order of risk:

1. **The 3-stage numeric path.** mHC → MLA-over-a-ring → 256-expert MoE, three
   times, written blind. The golden emits per-stage `attn_out` and `h_out`
   captures so a Spark-side compare can bisect to a stage; the ladder is
   `main_x` → `stage0.attn_out` → `stage0.h_out` → … → `x_pre_norm` →
   `logits_pre_markov`.
2. **`prefill_attn_compressed_vis`.** Written blind in Phase 1 and still never
   executed. DSpark is its second caller, with a materially different shape
   (5 queries, ring_len+5 keys, `n_comp = 0`, `ratio = 1`, padding rows with an
   empty key range). If it has a bug, Phase 2 and Phase 3 fail together.
3. **`dspark_capture.cu`.** New, never compiled.
4. **The EP=2 lane.** Both ranks must run the propose; the capture branch in
   `impl_b3_dspark.rs` deliberately has NO rank-0 early return. Never exercised.
5. **Ring/scheduler lockstep.** Any path that advances a sequence without a
   propose (serial-decode bootstrap, adaptive-speculation suspend,
   preemption/resume, a prefix-cache warm turn) leaves a ring gap. The design
   degrades rather than corrupts, and counts the gap:
   `RUST_LOG=spark_model=debug` prints `DSpark ring gap: …`. **Low measured
   acceptance on the Spark is far more likely to be a gap than an equation** —
   read that counter before touching the math.

   One gap is EXPECTED and is not a bug: on a cold turn the prefill fills the
   ring with prompt positions `0..P-1`, and the first propose drafts from
   position `P` — whose capture only exists after the next target step. The
   reference has the same shape (`forward_spec(..., 0)` fills the ring and
   returns without drafting; its first real spec pass follows a decode). Expect
   `gap: 1` once per sequence, and nothing after that.

### 7.6 First run

```bash
spark serve --model <vision-exp snapshot> --speculative             --max-batch-size 1 --max-seq-len 2048 --swap-space-gb 0
```

Expect these lines, in this order, before the first token:

```
DeepSeek-V4 native DSpark: 3 stages validated, block_size=5, …
DeepSeek-V4 native DSpark module loaded: 3 stages (MLA + mHC + MoE), EP rank …
DSpark target capture installed: layers [40, 41, 42], … rows …
DeepSeek-V4 native DSpark speculative decoding: ENABLED (3 stages, block_size=5, num_drafts=3)
Block drafter active: trained γ=5, driven at num_drafts=3 (verify K=4)
```

A missing `dspark_capture` module fails the boot audit by design (no
`[expected_absent]` entry) — that is the first thing to check if boot stops.

---

## See also

- [`DEEPSEEK_V4_0731_LOCAL_BRINGUP.md`](DEEPSEEK_V4_0731_LOCAL_BRINGUP.md) — the
  EP=2 procedure this document defers to, and the text-only checkpoints.
- [`GB10_DEPLOYMENT_GUIDE.md`](GB10_DEPLOYMENT_GUIDE.md) §2 — the serve matrix;
  this model is listed there as **UNVALIDATED — BRING-UP** and is not part of it.
- [`../tests/deepseek_v4_vision_compare.py`](../tests/deepseek_v4_vision_compare.py)
  — the reference compare script (`--mode selftest` documents its own contract).
- [`../AGENTS.md`](../AGENTS.md) — invariants, local checks, the certification and
  cross-hardware oracles.
