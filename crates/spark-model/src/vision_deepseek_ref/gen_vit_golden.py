#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""DeepSeek-V4-Flash-Vision-Exp ViT/aligner golden generator (unit U1.3).

Every activation this script emits comes out of DeepSeek's OWN `inference/vision.py`
(`ViT` / `Aligner` / `get_vision_cos_sin`), fetched at a pinned revision. Nothing is
re-derived here or in Rust: the Rust host reference in `vision_deepseek_ref.rs` is checked
against these arrays, so a passing test proves reference parity rather than
self-consistency.

`inference/vision.py` imports only torch, so — unlike U1.2's generator — no `kernel` stub
and no `model.py` import are needed. `ViT`/`Aligner` read exactly eight attributes off
their `args`, so a `types.SimpleNamespace` is enough.

Reference acquisition: `https://huggingface.co/<REPO>/raw/<REV>/inference/vision.py`, with a
local cache directory (`--cache`) as fallback. The fixture header records the revision, which
source served it, and a sha256 of the reference file, so a regeneration diff names exactly
which input moved.

# Determinism: RNG-free weights, so only outputs are committed

Parameters and inputs are filled from a name-salted hash stream, NOT an RNG, so Rust
regenerates them bit for bit and the fixture carries no weight data. The scheme is a verbatim
copy of the in-tree `gen_ds41_tiny_golden.py` / `layers::deepseek_v41_ref` scheme:

    fnv1a64(s)        = FNV-1a 64 over the UTF-8 bytes of s
                        (offset basis 0xCBF29CE484222325, prime 0x100000001B3)
    splitmix64(z)     = z += 0x9E3779B97F4A7C15
                        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                        return z ^ (z >> 31)                       (all mod 2**64)
    fixed_raw(n, i)   = splitmix64(fnv1a64(n) ^ ((i * 0x9E3779B97F4A7C15) mod 2**64))
    fixed_value(n, i, scale, offset)
                      = float32((2 * ((fixed_raw(n, i) >> 40) / 2**24) - 1) * scale + offset)
                        with the inner arithmetic in float64

Tensor names are the REAL checkpoint names (`vision.blocks.0.attn.wqkv.weight`,
`aligner.w1.bias`, ...), i.e. the torch parameter name under a `vision.`/`aligner.` prefix,
so the same names appear in `weights_meta`, in the Rust regeneration and in the checkpoint.
Fill rules, by parameter kind:

    RMSNorm weights (1-D, not a bias)      scale 0.10   offset 1.0
    biases                                 scale 0.02   offset 0.0
    aligner matrices                       scale 0.05   offset 0.0
    every other matrix                     scale 0.08   offset 0.0
    input patch pixels                     scale 1.00   offset 0.0

Those scales keep activations O(1), which is what makes the fixture's 1e-4 absolute bar a
real bar.

Run (this workspace has a venv with CPU torch; the system python3 is PEP 668-locked):

    /home/coder/.venvs/dsv4/bin/python3 \
        crates/spark-model/src/vision_deepseek_ref/gen_vit_golden.py

The script itself stays plain-`python3`-runnable (stdlib + torch only, no hard-coded
interpreter path), so it also runs on the Spark.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import types
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REV = "86f746b36186f0e567729a5c06a8c918caba82a9"
DEFAULT_CACHE = (
    "/tmp/claude-1000/-home-coder-dev-atlas/"
    "f9e8e1ea-da7e-4397-9cb7-8d96789896da/scratchpad/dsv4"
)
REF_FILE = "vision.py"

GENERATOR_INVOCATION = (
    "/home/coder/.venvs/dsv4/bin/python3 "
    "crates/spark-model/src/vision_deepseek_ref/gen_vit_golden.py"
)

# Tiny dims (unit U1.3 spec). head_dim = 16, rope_dim = 8, patch_in = 3*2*2 = 12.
DIMS = {
    "layers": 2,
    "dim": 32,
    "heads": 2,
    "inter": 40,
    "patch": 2,
    "ds": 3,
    "out_dim": 16,
    "rope_theta": 10000.0,
    "eps": 1e-6,
}

# (n_h, n_w): unshuffle padding on both axes, on neither (twice), and a single-row grid.
CASES = [(5, 7), (3, 3), (1, 4), (6, 6)]

TOLERANCE = {"abs": 1e-4, "rel": 1e-3}

MASK64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15


# --------------------------------------------------------------------------------------
# RNG-free filler (verbatim copy of the ds41 scheme)
# --------------------------------------------------------------------------------------

def splitmix64(z):
    z = (z + GOLD) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def fnv1a64(s):
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return h


def fixed_raw(name, i):
    return splitmix64(fnv1a64(name) ^ ((i * GOLD) & MASK64))


def f32(x):
    """Round a python float to float32 precision, exactly as torch's float32 cast does."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fixed_value(name, i, scale, offset):
    u = (fixed_raw(name, i) >> 40) / float(1 << 24)
    return f32((2.0 * u - 1.0) * scale + offset)


def fixed_list(name, n, scale, offset):
    return [fixed_value(name, i, scale, offset) for i in range(n)]


def checksum(values):
    """fp64 index-weighted checksum `sum(v[i] * (i + 1))` — the ds41 `ck`."""
    return float(sum(float(v) * (i + 1.0) for i, v in enumerate(values)))


# --------------------------------------------------------------------------------------
# reference acquisition
# --------------------------------------------------------------------------------------

def fetch_reference(cache_dir, workdir):
    """Fetch `inference/vision.py` (HF first, cache fallback) into `workdir`."""
    info = {"repo": REPO, "revision": REV, "source": "hf", "files": {}}
    url = f"https://huggingface.co/{REPO}/raw/{REV}/inference/{REF_FILE}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
    except Exception as exc:  # noqa: BLE001 - any network/HTTP failure falls back
        print(f"  ! HF fetch failed for {REF_FILE} ({exc}); using cache", file=sys.stderr)
        info["source"] = "cache"
        with open(os.path.join(cache_dir, "inference", REF_FILE), "rb") as handle:
            body = handle.read()
    with open(os.path.join(workdir, REF_FILE), "wb") as handle:
        handle.write(body)
    info["files"][REF_FILE] = hashlib.sha256(body).hexdigest()
    return info


# --------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------

def reference_args(d):
    """The eight attributes `ViT`/`Aligner` read off their `args`."""
    return types.SimpleNamespace(
        vision_patch_size=d["patch"],
        vision_dim=d["dim"],
        vision_n_heads=d["heads"],
        vision_inter_dim=d["inter"],
        vision_n_layers=d["layers"],
        vision_rope_theta=d["rope_theta"],
        vision_downsample_ratio=d["ds"],
        dim=d["out_dim"],
    )


def fill_rule(name, param):
    """(scale, offset) for a parameter, by kind. See the module docstring."""
    if name.endswith(".bias"):
        return 0.02, 0.0
    if param.dim() == 1:
        return 0.10, 1.0  # RMSNorm weight
    if name.startswith("aligner."):
        return 0.05, 0.0
    return 0.08, 0.0


def fill_parameters(torch, module, prefix, meta):
    """Fill every parameter of `module` from the hash stream, recording its rule."""
    for short, param in module.named_parameters():
        name = f"{prefix}{short}"
        scale, offset = fill_rule(name, param)
        values = fixed_list(name, param.numel(), scale, offset)
        with torch.no_grad():
            param.copy_(torch.tensor(values, dtype=torch.float32).view_as(param))
        meta[name] = {
            "n": param.numel(),
            "shape": list(param.shape),
            "scale": scale,
            "offset": offset,
            "dtype": "float32",
            "ck": checksum(values),
        }


# --------------------------------------------------------------------------------------
# captures
# --------------------------------------------------------------------------------------

def flat(tensor):
    return [float(v) for v in tensor.detach().to("cpu").float().reshape(-1).tolist()]


def build_case(torch, vision, vit, aligner, d, n_h, n_w):
    n = n_h * n_w
    patch_in = 3 * d["patch"] * d["patch"]
    name = f"patches.{n_h}x{n_w}"
    pixels = fixed_list(name, n * patch_in, 1.0, 0.0)
    patches = torch.tensor(pixels, dtype=torch.float32).view(n, 3, d["patch"], d["patch"])

    # `aligner.w1`'s input IS the unshuffled tensor and `aligner.w2`'s input IS the
    # post-GELU hidden state; neither has any other accessor on the reference.
    captured = {}

    def capture(key):
        return lambda _module, args: captured.__setitem__(key, args[0].detach().clone())

    hooks = [
        aligner.w1.register_forward_pre_hook(capture("unshuffle")),
        aligner.w2.register_forward_pre_hook(capture("aligner_hidden")),
    ]
    try:
        with torch.no_grad():
            cos, sin = vision.get_vision_cos_sin(n_h, n_w, vit.rope_dim, vit.rope_theta)
            patch_embed = vit.patch_embed(patches)
            vit_out = vit(patches, n_h, n_w)
            aligner_out = aligner(vit_out, n_h, n_w)
    finally:
        for hook in hooks:
            hook.remove()

    rope_dim = d["dim"] // d["heads"] // 2
    assert vit.rope_dim == rope_dim, (vit.rope_dim, rope_dim)
    assert tuple(cos.shape) == (n, 1, rope_dim), cos.shape
    assert tuple(vit_out.shape) == (n, d["dim"]), vit_out.shape
    assert tuple(patch_embed.shape) == (n, d["dim"]), patch_embed.shape
    n_llm_h, n_llm_w = -(-n_h // d["ds"]), -(-n_w // d["ds"])
    assert tuple(aligner_out.shape) == (n_llm_h * n_llm_w, d["out_dim"]), aligner_out.shape
    assert tuple(captured["aligner_hidden"].shape) == (n_llm_h * n_llm_w, d["out_dim"]), (
        captured["aligner_hidden"].shape
    )

    tensors = {
        "rope_cos": cos.squeeze(1),
        "rope_sin": sin.squeeze(1),
        "patch_embed": patch_embed,
        "vit_out": vit_out,
        "unshuffle": captured["unshuffle"],
        "aligner_hidden": captured["aligner_hidden"],
        "aligner_out": aligner_out,
    }
    case = {
        "n_h": n_h,
        "n_w": n_w,
        "n_llm_h": n_llm_h,
        "n_llm_w": n_llm_w,
        "patches_name": name,
        "patches_n": n * patch_in,
        "patches_ck": checksum(pixels),
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
    }
    for key, tensor in tensors.items():
        case[key] = flat(tensor)
    return case


# --------------------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------------------

_FLAT = re.compile(r"\[\s*((?:-?[\d.eE+]+\s*,\s*)*-?[\d.eE+]+\s*)\]")


def dumps_compact(obj):
    """`json.dumps(indent=1)` with every array of plain numbers collapsed onto one line."""
    text = json.dumps(obj, indent=1, sort_keys=False)
    while True:
        collapsed = _FLAT.sub(
            lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]", text
        )
        if collapsed == text:
            return collapsed
        text = collapsed


def write(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps_compact(obj))
        handle.write("\n")
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 ViT/aligner golden generator")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="directory holding inference/vision.py at the pinned revision")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "fixtures"))
    args = parser.parse_args()

    import torch

    torch.manual_seed(0)  # nothing here draws from it; set so the run is reproducible anyway
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001 - CPU SDPA may refuse; no RNG on this path
        print(f"  ! deterministic algorithms unavailable ({exc}); proceeding", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        header = fetch_reference(args.cache, workdir)
        header["generated"] = datetime.date.today().isoformat()
        sys.path.insert(0, workdir)
        import vision

        print(f"reference source: {header['source']} @ {REV}")
        d = DIMS
        ref_args = reference_args(d)
        vit = vision.ViT(ref_args).float().eval()
        aligner = vision.Aligner(ref_args).float().eval()

        weights_meta = {}
        fill_parameters(torch, vit, "vision.", weights_meta)
        fill_parameters(torch, aligner, "aligner.", weights_meta)

        fixture = {
            "reference": header,
            "generator": GENERATOR_INVOCATION,
            "dims": dict(d, head_dim=d["dim"] // d["heads"],
                         rope_dim=d["dim"] // d["heads"] // 2,
                         patch_in=3 * d["patch"] * d["patch"]),
            "tolerance": TOLERANCE,
            "rng_probe": {
                "name": "probe",
                "scale": 1.0,
                "offset": 0.0,
                "raw": [fixed_raw("probe", i) for i in range(8)],
                "values": fixed_list("probe", 8, 1.0, 0.0),
            },
            "weights_meta": weights_meta,
            "cases": [build_case(torch, vision, vit, aligner, d, n_h, n_w)
                      for n_h, n_w in CASES],
        }
        write(os.path.join(args.out, "vit_tiny_golden.json"), fixture)


if __name__ == "__main__":
    main()
