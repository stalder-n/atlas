#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Regenerate `fixtures/dspark_tiny_golden.json` from DeepSeek's own reference.

Runs `Transformer.forward` + `Transformer.forward_spec` from
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp `inference/model.py` at revision
86f746b36186f0e567729a5c06a8c918caba82a9, on a TINY config (dim 64, 6 layers,
2 experts, γ=5), and records every intermediate the Atlas DSpark head has to
reproduce.

Two deliberate substitutions, both recorded in the fixture header:

* `sys.modules["kernel"] = dsv4_ref_shims` — `kernel.py` is tilelang and
  cannot import on a CPU host. The shim is written from that file's source.
* `act_quant` → identity. `DSparkAttention.forward` round-trips the KV latent
  through FP8 to simulate its cache; Atlas's DSpark ring is BF16, so the FP8
  emulation is not the oracle for it.

Weights are deterministic (`fixed(name, i)`, splitmix64 — the
`deepseek_v41_ref` scheme), so the fixture commits only OUTPUTS: each capture
as `{shape, n, stride, ck, data}`, `ck` being the fp64 index-weighted checksum
over the whole tensor and `data` a prime-strided sample.

Run from the repository root:

    /home/coder/.venvs/dsv4/bin/python3 \\
        crates/spark-model/src/layers/deepseek_v4_dspark/gen_dspark_tiny_golden.py \\
        --ref <path to the reference `inference/` directory>

`--ref` defaults to the session scratchpad cache and falls back to
`$DSV4_REF`. The script itself is plain-`python3` runnable given torch + numpy.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = "86f746b36186f0e567729a5c06a8c918caba82a9"
REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
DEFAULT_REF = (
    "/tmp/claude-1000/-home-coder-dev-atlas/"
    "f9e8e1ea-da7e-4397-9cb7-8d96789896da/scratchpad/dsv4/inference"
)

MASK64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15


# ------------------------------------------------------------------ rng
def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return h


def fixed(name: str, n: int, scale: float, offset: float = 0.0) -> torch.Tensor:
    """`z = splitmix64(salt ^ (i * GOLD)); u = (z >> 40) / 2^24; (2u-1)*scale + offset`."""
    salt = np.uint64(fnv1a64(name))
    i = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = salt ^ (i * np.uint64(GOLD))
        z = z + np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    return torch.from_numpy(((2.0 * u - 1.0) * scale + offset).astype(np.float32))


def rule(name: str, p: torch.Tensor):
    """(scale, offset) by NAME, so the init is reproducible from `weights_meta` alone."""
    if name.endswith("attn_sink"):
        return 0.1, 0.0
    if name.endswith(("hc_attn_scale", "hc_ffn_scale", "hc_head_scale")):
        return 0.05, 1.0
    if name.endswith(("hc_attn_base", "hc_ffn_base", "hc_head_base")):
        return 0.1, 0.0
    if name.endswith(("gate.bias", "gate.bias_vl")):
        return 0.05, 0.0
    if name.endswith("norm.weight") or name.split(".")[-1].startswith("norm"):
        return 0.05, 1.0
    if p.dim() >= 2:
        return float(p.shape[-1]) ** -0.5, 0.0
    return 0.05, 0.0


def init_all(model: torch.nn.Module) -> dict:
    meta = {}
    for name, p in model.named_parameters():
        if p.dtype == torch.int32:
            # `Gate.tid2eid` (hash routing): deterministic expert ids.
            v = torch.from_numpy(
                (fixed(name, p.numel(), 1.0, 1.0).numpy() * 1e6).astype(np.int64)
                % max(1, int(FIX["n_routed_experts"]))
            ).to(torch.int32)
            with torch.no_grad():
                p.copy_(v.view(p.shape))
            meta[name] = dict(shape=list(p.shape), n=p.numel(), kind="int")
            continue
        scale, offset = rule(name, p)
        v = fixed(name, p.numel(), scale, offset).view(p.shape)
        with torch.no_grad():
            p.copy_(v.to(p.dtype))
        meta[name] = dict(
            shape=list(p.shape),
            n=p.numel(),
            scale=scale,
            offset=offset,
            dtype=str(p.dtype).removeprefix("torch."),
            ck=ck(p.detach().float()),
        )
    return meta


# ------------------------------------------------------------------ emit
def ck(t: torch.Tensor) -> float:
    f = t.detach().reshape(-1).to(torch.float64)
    i = torch.arange(f.numel(), dtype=torch.float64)
    return float((f * (i + 1.0)).sum())


def _next_prime(k: int) -> int:
    k = max(2, k)
    while any(k % d == 0 for d in range(2, int(k**0.5) + 1)):
        k += 1
    return k


# Captures the Rust host reference consumes as INPUT are stored whole; the rest
# are prime-strided samples plus the whole-tensor checksum.
FULL = {"x_pre_norm", "logits_pre_markov", "confidence", "output_ids", "markov_embed"}


def emit(name: str, t: torch.Tensor, is_int: bool = False) -> dict:
    n = t.numel()
    cap = 64 if is_int else 12
    stride = 1 if (name in FULL or n <= cap) else _next_prime(-(-n // cap))
    flat = t.detach().reshape(-1)
    if is_int:
        return dict(
            shape=list(t.shape),
            n=n,
            stride=stride,
            ck=ck(flat.to(torch.float64)),
            data=[int(x) for x in flat.tolist()[::stride]],
        )
    return dict(
        shape=list(t.shape),
        n=n,
        stride=stride,
        ck=ck(flat.float()),
        data=[float(x) for x in flat.float().tolist()[::stride]],
    )


REC: dict[str, dict] = {}


def rec(name: str, t: torch.Tensor, is_int: bool = False):
    REC[name] = emit(name, t.detach().clone().cpu(), is_int)


# ------------------------------------------------------------------ config
FIX = dict(
    max_batch_size=1,
    max_seq_len=64,
    temperature=0,
    dtype="bf16",
    scale_fmt=None,
    expert_dtype=None,
    scale_dtype="fp32",
    vocab_size=32,
    dim=64,
    moe_inter_dim=24,
    n_layers=6,
    n_hash_layers=1,
    n_mtp_layers=3,
    n_heads=4,
    n_routed_experts=2,
    n_shared_experts=1,
    n_activated_experts=2,
    q_lora_rank=16,
    head_dim=32,
    rope_head_dim=8,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=16,
    window_size=8,
    compress_ratios=(0,) * 9,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    dspark_block_size=5,
    dspark_noise_token_id=3,
    dspark_target_layer_ids=(3, 4, 5),
    dspark_markov_rank=8,
    vision_n_layers=0,
)
PROMPT_LEN = 11  # > window_size (8), so the ring WRAPS: the `cutoff` split runs.


def patch(cls, attr, wrapper):
    orig = getattr(cls, attr)
    setattr(cls, attr, functools.wraps(orig)(lambda *a, **k: wrapper(orig, *a, **k)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.environ.get("DSV4_REF", DEFAULT_REF))
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "dspark_tiny_golden.json"))
    args = ap.parse_args()
    if not os.path.isdir(args.ref):
        print(f"reference not found: {args.ref} (pass --ref or set DSV4_REF)", file=sys.stderr)
        return 2

    sys.path.insert(0, HERE)
    sys.path.insert(0, args.ref)
    import dsv4_ref_shims  # noqa: E402

    sys.modules["kernel"] = dsv4_ref_shims
    import model as ref  # noqa: E402

    # 🔴 Recorded in the header: the reference simulates its FP8 KV cache with
    # an in-place quant/dequant round trip. Atlas's DSpark ring is BF16, so the
    # FP8 emulation is not the oracle for it.
    ref.act_quant = lambda x, *a, **k: x

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.bfloat16)
    margs = ref.ModelArgs(**FIX)
    m = ref.Transformer(margs)
    weights_meta = init_all(m)
    for _, buf in m.named_buffers():
        if buf.dtype.is_floating_point and buf.dim() >= 2:
            buf.zero_()

    # ── hooks ──
    def forward_embed(orig, self, main_hidden, input_ids):
        rec("main_hidden_in", main_hidden)
        x, main_x = orig(self, main_hidden, input_ids)
        rec("block_x_after_hc_expand", x)
        rec("main_x", main_x)
        return x, main_x

    def dspark_attn(orig, self, x, start_pos, main_x):
        out = orig(self, x, start_pos, main_x)
        if start_pos > 0:
            rec(f"stage{self.layer_id - FIX['n_layers']}.attn_out", out)
        return out

    def dspark_block(orig, self, x, start_pos, input_ids, main_x):
        out = orig(self, x, start_pos, input_ids, main_x)
        if start_pos > 0:
            rec(f"stage{self.layer_id - FIX['n_layers']}.h_out", out)
        return out

    def head_forward(orig, self, x, full_logits=False):
        out = orig(self, x, full_logits)
        if full_logits and out.dim() == 3 and out.size(1) == FIX["dspark_block_size"]:
            # Cloned: `forward_head` adds the bigram bias into this tensor IN
            # PLACE, so the reference held later is the BIASED one.
            rec("logits_pre_markov", out.clone())
        return out

    def confidence(orig, self, hidden, markov_embed):
        # `hidden` is the PRE-`norm` hc_head output — the `x` the Atlas head
        # must feed its confidence dot, not the normed rows.
        rec("x_pre_norm", hidden)
        rec("markov_embed", markov_embed)
        out = orig(self, hidden, markov_embed)
        rec("confidence", out)
        return out

    patch(ref.DSparkBlock, "forward_embed", forward_embed)
    patch(ref.DSparkAttention, "forward", dspark_attn)
    patch(ref.DSparkBlock, "forward", dspark_block)
    patch(ref.ParallelHead, "forward", head_forward)
    patch(ref.DSparkConfidenceHead, "forward", confidence)

    # ── prompt: target forward, then the ring-fill spec pass ──
    ids = torch.arange(PROMPT_LEN + 1, dtype=torch.long).remainder(FIX["vocab_size"]).view(1, -1)
    rec("input_ids", ids, is_int=True)
    out_ids, _, main_hidden = m(ids[:, :PROMPT_LEN])
    rec("prompt_main_hidden", main_hidden)
    rec("prompt_output_ids", out_ids, is_int=True)
    m.forward_spec(out_ids, main_hidden, 0)
    for s, stage in enumerate(m.mtp):
        rec(f"stage{s}.ring_after_prefill", stage.attn.kv_cache[:1, : FIX["window_size"]])

    # ── one decode step, then the real spec pass ──
    out_ids, _, main_hidden = m(ids[:, PROMPT_LEN : PROMPT_LEN + 1], PROMPT_LEN)
    rec("decode_main_hidden", main_hidden)
    rec("decode_output_ids", out_ids, is_int=True)
    spec_ids, spec_logits, spec_conf = m.forward_spec(out_ids, main_hidden, PROMPT_LEN)
    rec("output_ids", spec_ids, is_int=True)
    rec("logits_post_markov", spec_logits)
    for s, stage in enumerate(m.mtp):
        rec(f"stage{s}.ring_after_decode", stage.attn.kv_cache[:1, : FIX["window_size"]])

    doc = dict(
        _generator=dict(
            script=(
                "crates/spark-model/src/layers/deepseek_v4_dspark/gen_dspark_tiny_golden.py"
            ),
            invocation=(
                "/home/coder/.venvs/dsv4/bin/python3 "
                "crates/spark-model/src/layers/deepseek_v4_dspark/gen_dspark_tiny_golden.py"
            ),
            reference=dict(repo=REPO, revision=REVISION, entry="inference/model.py"),
            substitutions=[
                "sys.modules['kernel'] = dsv4_ref_shims (tilelang cannot import on a CPU host)",
                "act_quant -> identity: the reference round-trips the KV latent through FP8 to "
                "simulate its cache; Atlas's DSpark ring is BF16, so that emulation is not the "
                "oracle for it",
            ],
            rng=(
                "fixed(name, i): z = splitmix64(fnv1a64(name) ^ (i * 0x9E3779B97F4A7C15)); "
                "u = (z >> 40) / 2^24; value = f32((2u - 1) * scale + offset)"
            ),
            torch=torch.__version__,
        ),
        config=dict(FIX, prompt_len=PROMPT_LEN),
        weights_meta=weights_meta,
        captures=REC,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {args.out} ({len(REC)} captures)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
