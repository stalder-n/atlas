#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""DeepSeek-V4-Flash-Vision-Exp image-layout golden generator (unit U1.2).

Every number this script emits is produced by DeepSeek's OWN reference code, fetched at a
pinned revision — nothing is re-derived in Rust or here, with one documented exception (the
~10-line sizing prologue of `load_image`, which the reference embeds in a function that opens
a real file; it is copied verbatim below and cross-checked against the real `load_image` on
in-memory PNGs).

Reference files (`inference/{image_processor,vision,model}.py`) are fetched from
`https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/<REV>/...` and fall back
to a local cache directory (`--cache`). Each fixture header records the revision, which source
was used, and a sha256 per reference file, so a regeneration diff names exactly which input
moved.

`inference/model.py` does `from kernel import ...`, and `kernel.py` needs tilelang (not
installed, GPU-only). A six-name stub module is installed in `sys.modules["kernel"]` before
the import; every stub raises if called. This mirrors the in-tree `ds41_ref_shims.py`
precedent. Nothing this script calls in `model.py` touches those names.

Outputs (beside this script, in `fixtures/`):

  geometry.json    plan_image / grid_tokens / solve_resize_ratio sizing goldens
  block.json       build_image_block token types + aligner permutation, all 12x12 grids
  visibility.json  get_image_visible + get_window_topk_idxs_visible over synthetic streams
  resample.json    PIL Image.resize(BICUBIC) and ImageOps.pad outputs on a gradient image

Run (this workspace has a venv with CPU torch, Pillow, numpy; the system python3 is
PEP 668-locked and has none of them):

    /home/coder/.venvs/dsv4/bin/python3 \
        crates/spark-model/src/vision_deepseek/gen_layout_goldens.py

The script itself stays plain-`python3`-runnable (stdlib + torch/PIL/numpy only, no hard-coded
interpreter path), so it also runs on the Spark.
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import re
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
REF_FILES = ("image_processor.py", "vision.py", "model.py")

# The reference's own config.json values for this checkpoint. Pinned here so the goldens
# state the geometry they were generated under; the Rust side reads them back out.
GEOMETRY = {
    "patch_size": 14,        # vision_patch_size
    "downsample_ratio": 3,   # vision_downsample_ratio
    "max_n_token": 384,      # vision_max_n_token
    "min_pixels": 147456,    # vision_min_pixels
    "max_wh_ratio": 8,       # vision_max_wh_ratio
}
VOCAB_SIZE = 129280
SLIDING_WINDOW = 128

# The deterministic source image for the resampling goldens. Duplicated VERBATIM in
# `vision_deepseek.rs::testutil::gradient_rgb` — change one, change both.
#   r = (7x + 13y) % 256 ,  g = (x*x + 3y) % 256 ,  b = (x ^ y) % 256
GRADIENT_FORMULA = "gradient_rgb(x,y) = ((7x+13y)%256, (x*x+3y)%256, (x^y)%256)"


def gradient_rgb(w, h):
    import numpy as np

    x = np.arange(w, dtype=np.int64)[None, :]
    y = np.arange(h, dtype=np.int64)[:, None]
    r = (7 * x + 13 * y) % 256
    g = (x * x + 3 * y) % 256
    b = (x ^ y) % 256
    return np.stack(np.broadcast_arrays(r, g, b), axis=-1).astype("uint8")


# --------------------------------------------------------------------------------------
# reference acquisition
# --------------------------------------------------------------------------------------

def fetch_reference(cache_dir, workdir):
    """Fetch the reference files (HF first, cache fallback) into `workdir`.

    Returns the header dict recorded in every fixture.
    """
    info = {"repo": REPO, "revision": REV, "source": "hf", "files": {}}
    for name in REF_FILES:
        url = f"https://huggingface.co/{REPO}/raw/{REV}/inference/{name}"
        body = None
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
        except Exception as exc:  # noqa: BLE001 - any network/HTTP failure falls back
            print(f"  ! HF fetch failed for {name} ({exc}); using cache", file=sys.stderr)
            info["source"] = "cache"
            with open(os.path.join(cache_dir, "inference", name), "rb") as handle:
                body = handle.read()
        with open(os.path.join(workdir, name), "wb") as handle:
            handle.write(body)
        info["files"][name] = hashlib.sha256(body).hexdigest()
    return info


def install_kernel_stub():
    """`model.py` imports six tilelang kernels at module scope. Stub them; none is called."""
    stub = types.ModuleType("kernel")
    for name in (
        "act_quant",
        "fp4_act_quant",
        "fp8_gemm",
        "fp4_gemm",
        "sparse_attn",
        "hc_split_sinkhorn",
    ):
        def _unavailable(*_a, _name=name, **_k):
            raise RuntimeError(f"kernel.{_name} is a golden-generator stub and is never called")

        setattr(stub, name, _unavailable)
    sys.modules["kernel"] = stub


# --------------------------------------------------------------------------------------
# geometry.json
# --------------------------------------------------------------------------------------

# Verbatim copy of `image_processor.py::load_image` lines between `image.size` and
# `safe_resize` — the only reference logic this script restates, because the reference embeds
# it in a function that opens a real image file. Cross-checked against the real `load_image`
# in `crosscheck_load_image` below.
def load_image_prologue(width, height, g):
    import math

    if g["max_wh_ratio"] is not None and width > height * g["max_wh_ratio"]:
        width = height * g["max_wh_ratio"]
    if 0 < width * height < g["min_pixels"]:
        ratio = (g["min_pixels"] / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    p = g["patch_size"]
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return height, width, best_height, best_width


PLAN_CASES = [
    # (w, h, note)
    (1, 1, "degenerate 1x1, far below min_pixels -> upscale"),
    (64, 64, "small square below min_pixels -> upscale"),
    (384, 384, "square just below min_pixels -> upscale"),
    (383, 384, "near-square below min_pixels"),
    (3, 7000, "extremely tall: max_w_float < 1 branch"),
    (4, 6000, "extremely tall, second max_w_float < 1 case"),
    (2000, 100, "w = 20h -> squash (plain resize, no pad)"),
    (1600, 200, "w = 8h exactly -> squash boundary"),
    (1599, 200, "w = 8h - 1 -> NOT squash (near-squash)"),
    (1601, 200, "w = 8h + 1 -> squash"),
    (8, 1, "tiny wide, w = 8h -> squash and upscale"),
    (20000, 20, "extreme wide -> clamp then squash"),
    (100, 1000, "tall 1:10"),
    (1000, 100, "wide 10:1 -> squash"),
    (1920, 1080, "1080p landscape"),
    (1080, 1920, "1080p portrait"),
    (640, 480, "VGA"),
    (480, 640, "VGA portrait"),
    (1024, 768, "XGA"),
    (768, 1024, "XGA portrait"),
    (800, 600, "SVGA"),
    (1280, 720, "720p"),
    (1512, 982, "odd laptop screenshot"),
    (4000, 3000, "12 MP photo -> safe_resize budget loop"),
    (8000, 6000, "48 MP photo -> safe_resize budget loop"),
    (12000, 1500, "panorama 8:1 -> squash boundary + budget loop"),
    (12000, 1501, "panorama just under 8:1 -> pad + budget loop"),
    (5000, 5000, "large square -> budget loop"),
    (16384, 16384, "max-side square"),
    (333, 777, "odd tall"),
    (777, 333, "odd wide"),
    (14, 14, "exactly one patch before upscale"),
    (15, 13, "sub-patch, non-multiple of patch_size"),
    (28, 14, "two patches wide"),
    (100, 100, "small square"),
    (200, 150, "4:3 small"),
    (150, 200, "3:4 small"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4K"),
    (2048, 2048, "2K square"),
    (997, 991, "coprime near-square"),
    (1231, 457, "coprime wide"),
    (457, 1231, "coprime tall"),
    (6000, 800, "7.5:1 wide, just under squash"),
    (6400, 800, "8:1 wide, exactly squash"),
    (31, 4001, "very tall thin"),
]

# Direct `solve_resize_ratio` calls. `plan_image` cannot reach the `max_h_float < 2` branch
# (load_image clamps width <= max_wh_ratio*height first, so r >= 1/8 and max_h_float ~ 6.8),
# so that branch is pinned by calling the reference function directly.
SOLVE_CASES = [
    # (h, w, max_n_token, note)
    (100, 10000, 381, "r = 0.01 -> max_h_float < 2 branch"),
    (10, 10000, 381, "r = 0.001 -> max_h_float < 2 branch"),
    (50, 20000, 381, "extreme ratio -> max_h_float < 2 branch"),
    (10000, 100, 381, "r = 100 -> max_w_float < 1 branch"),
    (7000, 3, 381, "the plan_image tall case, direct"),
    (768, 1024, 381, "ordinary landscape, else branch"),
    (1024, 768, 381, "ordinary portrait, else branch"),
    (600, 600, 381, "square, else branch"),
    (600, 600, 200, "square with a smaller budget"),
    (600, 600, 64, "square with a small budget"),
    (600, 600, 20, "square with a tiny budget"),
    (100, 10000, 100, "max_h_float < 2 with a small budget"),
    (1, 100, 6, "max_h_float < 2 with budget 6 -> the `assert max_w > 1` fires"),
    (1, 100, 7, "max_h_float < 2 with budget 7 -> the `assert max_w > 1` fires"),
    (1, 100, 8, "max_h_float < 2 with budget 8 -> max_w == 2, the assert just holds"),
]


def build_geometry_fixture(ip, header):
    g = GEOMETRY
    grid_seen = set()
    grid_rows = []
    plan_rows = []
    for (w, h, note) in PLAN_CASES:
        height, width, best_h, best_w = load_image_prologue(w, h, g)
        n_llm_h, n_llm_w, best_h, best_w = ip.safe_resize(
            height, width, best_h, best_w, g["patch_size"], g["downsample_ratio"], g["max_n_token"]
        )
        _, _, num_tokens = ip.grid_tokens(
            best_h, best_w, g["patch_size"], g["downsample_ratio"]
        )
        # The pre-safe_resize grid, so the fixture records whether the budget loop ran at
        # all (initial_num_tokens > max_n_token - (COMPRESS_PAD_TO - 1) <=> at least one
        # solve_resize_ratio iteration). Reference call, not a re-derivation.
        _, _, initial_num_tokens = ip.grid_tokens(
            *load_image_prologue(w, h, g)[2:], g["patch_size"], g["downsample_ratio"]
        )
        squash = w >= g["max_wh_ratio"] * h
        plan_rows.append(
            {
                "w": w,
                "h": h,
                # The `load_image` sizing prologue's output = `safe_resize`'s inputs, so the
                # Rust test can drive `safe_resize` directly as well as through `plan_image`.
                "pre_h": height,
                "pre_w": width,
                "pre_best_h": load_image_prologue(w, h, g)[2],
                "pre_best_w": load_image_prologue(w, h, g)[3],
                "best_w": best_w,
                "best_h": best_h,
                "n_vit_h": best_h // g["patch_size"],
                "n_vit_w": best_w // g["patch_size"],
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "squash": bool(squash),
                "num_tokens": num_tokens,
                "initial_num_tokens": initial_num_tokens,
                "note": note,
            }
        )
        grid_seen.add((best_h, best_w))

    # A fan of direct grid_tokens calls, including every (best_h, best_w) the plans produced
    # plus a dense sweep of small grids so the parity / alignment terms are pinned.
    p, ds = g["patch_size"], g["downsample_ratio"]
    for nh in range(1, 15):
        for nw in range(1, 15):
            grid_seen.add((nh * p * ds, nw * p * ds))
    for (best_h, best_w) in sorted(grid_seen):
        n_llm_h, n_llm_w, num_tokens = ip.grid_tokens(best_h, best_w, p, ds)
        grid_rows.append(
            {
                "best_h": best_h,
                "best_w": best_w,
                "patch": p,
                "ds": ds,
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "num_tokens": num_tokens,
            }
        )

    solve_rows = []
    for (h, w, budget, note) in SOLVE_CASES:
        r = h / w
        import math

        max_w_float = math.sqrt((budget - 2) / r + 0.25) - 0.5
        branch = (
            "max_w_float<1"
            if max_w_float < 1.0
            else ("max_h_float<2" if max_w_float * r < 2.0 else "else")
        )
        try:
            n_llm_h, n_llm_w, best_h, best_w, num_tokens = ip.solve_resize_ratio(h, w, p, ds, budget)
        except AssertionError:
            solve_rows.append(
                {"h": h, "w": w, "patch": p, "ds": ds, "max_n_token": budget,
                 "branch": branch, "assert_fails": True, "note": note}
            )
            continue
        solve_rows.append(
            {
                "h": h,
                "w": w,
                "patch": p,
                "ds": ds,
                "max_n_token": budget,
                "branch": branch,
                "assert_fails": False,
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "best_h": best_h,
                "best_w": best_w,
                "num_tokens": num_tokens,
                "note": note,
            }
        )

    branches = {row["branch"] for row in solve_rows}
    assert branches == {"max_w_float<1", "max_h_float<2", "else"}, branches
    assert any(r["assert_fails"] for r in solve_rows), "no case reaches `assert max_w > 1`"
    assert any(r["squash"] for r in plan_rows) and any(not r["squash"] for r in plan_rows)
    budget = g["max_n_token"] - 3
    looped = [r for r in plan_rows if r["initial_num_tokens"] > budget]
    assert len(looped) >= 8, f"only {len(looped)} plan cases iterate the safe_resize loop"
    assert len(plan_rows) >= 40, len(plan_rows)

    return {
        "reference": header,
        "generator": "python3 crates/spark-model/src/vision_deepseek/gen_layout_goldens.py",
        "geometry": g,
        "plan": plan_rows,
        "grid_tokens": grid_rows,
        "solve": solve_rows,
    }


def crosscheck_load_image(ip, header):
    """Run the REAL `ip.load_image` on in-memory PNGs and assert the prologue copy agrees."""
    import numpy as np
    from PIL import Image

    class Args:
        pass

    args = Args()
    args.vision_patch_size = GEOMETRY["patch_size"]
    args.vision_downsample_ratio = GEOMETRY["downsample_ratio"]
    args.vision_max_n_token = GEOMETRY["max_n_token"]
    args.vision_min_pixels = GEOMETRY["min_pixels"]
    args.vision_max_wh_ratio = GEOMETRY["max_wh_ratio"]

    checked = []
    for (w, h) in [(64, 64), (200, 150), (1024, 768), (2000, 100), (97, 61), (3, 700)]:
        buf = io.BytesIO()
        Image.fromarray(gradient_rgb(w, h)).save(buf, format="PNG")
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": buf.getvalue()}, args)
        height, width, best_h, best_w = load_image_prologue(w, h, GEOMETRY)
        sh, sw, sbh, sbw = ip.safe_resize(
            height, width, best_h, best_w,
            GEOMETRY["patch_size"], GEOMETRY["downsample_ratio"], GEOMETRY["max_n_token"],
        )
        assert (sh, sw) == (n_llm_h, n_llm_w), (w, h, sh, sw, n_llm_h, n_llm_w)
        assert (sbh // GEOMETRY["patch_size"], sbw // GEOMETRY["patch_size"]) == (n_vit_h, n_vit_w)
        assert tuple(patches.shape) == (n_vit_h * n_vit_w, 3,
                                        GEOMETRY["patch_size"], GEOMETRY["patch_size"])
        lo = patches.float().min().item()
        hi = patches.float().max().item()
        assert -1.0 <= lo <= hi <= 1.0, (lo, hi)
        checked.append((w, h))
    print(f"  load_image cross-check OK on {checked}")


# --------------------------------------------------------------------------------------
# block.json
# --------------------------------------------------------------------------------------

def build_block_fixture(ip, header):
    grids = []
    for nh in range(1, 13):
        for nw in range(1, 13):
            per_start = {}
            for sp in range(4):
                types, perm = ip.build_image_block(nh, nw, sp)
                per_start[str(sp)] = "".join(str(int(v)) for v in types.tolist())
            _, perm0 = ip.build_image_block(nh, nw, 0)
            _, _, num_tokens = ip.grid_tokens(
                nh * GEOMETRY["patch_size"] * GEOMETRY["downsample_ratio"],
                nw * GEOMETRY["patch_size"] * GEOMETRY["downsample_ratio"],
                GEOMETRY["patch_size"],
                GEOMETRY["downsample_ratio"],
            )
            grids.append(
                {
                    "h": nh,
                    "w": nw,
                    "perm": [int(v) for v in perm0.tolist()],
                    "rows": nh + nh % 2,
                    "row_len": nw + 1,
                    "pad_last": (nh + nh % 2) // 2 * (nw + 1) % 2 * 2,
                    "types": per_start,
                    "num_tokens": num_tokens,
                }
            )

    large = []
    for (nh, nw) in [(1, 1), (2, 3), (3, 3), (5, 8), (12, 12), (7, 2)]:
        for sp in [4, 7, 129, 1000, 2049, 4095]:
            types, perm = ip.build_image_block(nh, nw, sp)
            large.append(
                {
                    "h": nh,
                    "w": nw,
                    "start_pos": sp,
                    "types": "".join(str(int(v)) for v in types.tolist()),
                    "perm": [int(v) for v in perm.tolist()],
                }
            )

    # The expansion golden: run the reference's own `prepare_vl_inputs` expansion loop
    # (the three lines that are not tokenizer/image I/O) over a synthetic token stream.
    expansions = []
    for (tokens, gridspec, note) in [
        ([10, 999, 11, 999, 12], [(2, 3), (4, 4)], "two images, text before/between/after"),
        ([999], [(1, 1)], "image only"),
        ([999, 999], [(3, 5), (3, 5)], "two adjacent images"),
        ([1, 2, 3, 4, 5, 6, 7, 999], [(6, 7)], "image at the end, start_pos = 7"),
        ([999, 1, 2, 3], [(12, 12)], "image at position 0"),
        ([1, 2, 999, 3, 999, 4, 999, 5], [(1, 4), (2, 2), (5, 1)], "three images"),
    ]:
        out = []
        blocks = []
        it = iter(gridspec)
        for tok in tokens:
            if tok != 999:
                out.append(tok)
                continue
            nh, nw = next(it)
            types, perm = ip.build_image_block(nh, nw, len(out))
            blocks.append(
                {
                    "start": len(out),
                    "h": nh,
                    "w": nw,
                    "types": "".join(str(int(v)) for v in types.tolist()),
                    "perm": [int(v) for v in perm.tolist()],
                }
            )
            out += (VOCAB_SIZE + types).tolist()
        expansions.append(
            {
                "note": note,
                "tokens": tokens,
                "placeholder": 999,
                "grids": [list(x) for x in gridspec],
                "expanded": [int(v) for v in out],
                "blocks": blocks,
            }
        )

    return {
        "reference": header,
        "generator": "python3 crates/spark-model/src/vision_deepseek/gen_layout_goldens.py",
        "vocab_size": VOCAB_SIZE,
        "compress_pad_to": 4,
        "grids": grids,
        "large_start_pos": large,
        "expansions": expansions,
    }


# --------------------------------------------------------------------------------------
# visibility.json
# --------------------------------------------------------------------------------------

def make_span(ip, nh, nw, start_pos):
    types, _ = ip.build_image_block(nh, nw, start_pos)
    return [VOCAB_SIZE + int(v) for v in types.tolist()]


def build_visibility_fixture(ip, M, header):
    import torch

    sys.path_hooks  # noqa: B018 - keep the import of `model` explicit below
    import model as M_mod

    streams = []

    def add(name, tokens, note):
        ids = torch.tensor([tokens], dtype=torch.int64)
        left, right = M_mod.get_image_visible(ids, VOCAB_SIZE, M)
        topk = M_mod.get_window_topk_idxs_visible(
            SLIDING_WINDOW, len(tokens), left, right, M
        )
        row_sets = []
        lo_list, hi_list, counts = [], [], []
        contiguous = True
        for t in range(len(tokens)):
            vals = sorted({int(v) for v in topk[0, t].tolist() if int(v) >= 0})
            assert vals, f"{name}: query {t} sees nothing"
            lo_list.append(vals[0])
            hi_list.append(vals[-1])
            counts.append(len(vals))
            if vals != list(range(vals[0], vals[-1] + 1)):
                contiguous = False
            row_sets.append(vals)
        entry = {
            "name": name,
            "note": note,
            "len": len(tokens),
            "tokens": tokens,
            "left": [int(v) for v in left[0].tolist()],
            "right": [int(v) for v in right[0].tolist()],
            "lo": lo_list,
            "hi": hi_list,
            "visible_counts": counts,
            "visible_contiguous": contiguous,
            "has_image_span": any(t >= VOCAB_SIZE for t in tokens),
        }
        if len(tokens) <= 64:
            entry["visible_sets"] = row_sets
        streams.append(entry)

    add("text_only", list(range(5, 45)), "no sentinels at all -> plain causal window")
    add("image_at_zero", make_span(ip, 2, 3, 0) + [7, 8, 9], "span starts at position 0")
    add(
        "image_mid",
        [1, 2, 3] + make_span(ip, 2, 2, 3) + [4, 5, 6, 7],
        "text, one image, text",
    )
    add(
        "two_adjacent",
        [1, 2] + make_span(ip, 1, 2, 2) + make_span(ip, 2, 1, 2 + len(make_span(ip, 1, 2, 2))) + [3],
        "two images with no text between them",
    )
    add(
        "image_at_end",
        [1, 2, 3, 4] + make_span(ip, 2, 2, 4),
        "span runs to the last token -> unterminated on the right? no, END is last",
    )
    long_pre = list(range(100, 100 + 700))
    long_stream = long_pre + make_span(ip, 3, 4, len(long_pre)) + list(range(200, 240))
    add("long_640", long_stream, "seqlen ~ 800 > window (128); one image")
    pre = list(range(1000, 1000 + 900))
    s1 = make_span(ip, 4, 5, len(pre))
    mid = list(range(2000, 2000 + 400))
    s2 = make_span(ip, 2, 6, len(pre) + len(s1) + len(mid))
    tail = list(range(3000, 3000 + 300))
    total = pre + s1 + mid + s2 + tail
    assert len(total) >= 2048 - 400, len(total)
    total = total + list(range(4000, 4000 + (2048 - len(total))))
    add("prompt_2048", total, "2048-token prompt with two images; seqlen > window+max_image_tokens")

    # SYNTHETIC MALFORMED stream: a 600-sentinel span. No real image can produce one
    # (safe_resize caps a block at 381 tokens), so this is the only way the `left`/`right`
    # clamps and the `lo + W + M - 1` cap are pinned by the reference rather than by a
    # hand reading of it. Explicitly labelled so nobody mistakes it for a real layout.
    mal = [1, 2, 3] + [VOCAB_SIZE + 0] + [VOCAB_SIZE + 2] * 600 + [VOCAB_SIZE + 4] + [4, 5]
    add("synthetic_malformed_600", mal, "SYNTHETIC: 600-token span, exercises the left/right clamps and the lo+W+M-1 cap")

    assert any(s["name"] == "synthetic_malformed_600" for s in streams)
    assert all(s["visible_contiguous"] for s in streams), "a visible set is not an interval"

    return {
        "reference": header,
        "generator": "python3 crates/spark-model/src/vision_deepseek/gen_layout_goldens.py",
        "window": SLIDING_WINDOW,
        "max_image_tokens": M,
        "vocab_size": VOCAB_SIZE,
        "streams": streams,
    }


# --------------------------------------------------------------------------------------
# resample.json
# --------------------------------------------------------------------------------------

RESIZE_CASES = [
    (97, 61, 28, 14, "both axes down, non-integer ratio"),
    (28, 14, 42, 28, "both axes up"),
    (40, 30, 40, 15, "height only (width equal -> horizontal pass skipped)"),
    (40, 30, 20, 30, "width only (height equal -> vertical pass skipped)"),
    (33, 17, 14, 14, "non-integer ratio both axes"),
    (200, 150, 14, 14, "extreme downscale: the support-widening term dominates"),
    (28, 28, 28, 28, "identity -> PIL's copy shortcut"),
    (17, 11, 34, 13, "up in w, up in h by a non-integer factor"),
    (13, 29, 41, 14, "mixed up/down"),
    (64, 64, 21, 21, "square downscale"),
]

PAD_CASES = [
    (97, 61, 42, 42, "wider than target -> pads vertically"),
    (61, 97, 42, 42, "taller than target -> pads horizontally"),
    (40, 30, 40, 30, "already the target aspect and size -> no pad, identity"),
    (20, 15, 40, 30, "same aspect, pure upscale -> no pad"),
    (33, 17, 28, 28, "odd source -> pads vertically, odd offset (banker's rounding)"),
    (17, 33, 28, 28, "odd source -> pads horizontally"),
    (50, 50, 28, 14, "square into a 2:1 target -> pads horizontally"),
    (50, 50, 14, 28, "square into a 1:2 target -> pads vertically"),
]


def build_resample_fixture(header):
    import numpy as np
    from PIL import Image, ImageOps

    resize_rows = []
    for (sw, sh, dw, dh, note) in RESIZE_CASES:
        src = Image.fromarray(gradient_rgb(sw, sh))
        out = src.resize((dw, dh), Image.BICUBIC)
        resize_rows.append(
            {
                "sw": sw, "sh": sh, "dw": dw, "dh": dh, "note": note,
                "rgb": np.asarray(out, dtype=np.uint8).reshape(-1).tolist(),
            }
        )

    pad_rows = []
    for (sw, sh, dw, dh, note) in PAD_CASES:
        src = Image.fromarray(gradient_rgb(sw, sh))
        out = ImageOps.pad(src, (dw, dh), color=(127, 127, 127))
        assert out.size == (dw, dh), (out.size, dw, dh)
        pad_rows.append(
            {
                "sw": sw, "sh": sh, "dw": dw, "dh": dh, "note": note,
                "rgb": np.asarray(out, dtype=np.uint8).reshape(-1).tolist(),
            }
        )

    # Python's round() is banker's rounding, and ImageOps.contain uses it for the
    # contained dimension and ImageOps.pad for the paste offset. Pinned as a table.
    round_rows = [[x, round(x)] for x in
                  [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 4.5,
                   0.49999999, 0.5000001, 1.4999999, 12.5, 13.5, 20.5]]

    return {
        "reference": header,
        "generator": "python3 crates/spark-model/src/vision_deepseek/gen_layout_goldens.py",
        "source": GRADIENT_FORMULA,
        "fill": [127, 127, 127],
        "resize": resize_rows,
        "pad": pad_rows,
        "py_round_half_even": round_rows,
    }


# --------------------------------------------------------------------------------------

_FLAT = re.compile(r"\[\s*((?:-?[\d.eE+]+\s*,\s*)*-?[\d.eE+]+\s*)\]")


def dumps_compact(obj):
    """`json.dumps(indent=1)`, but every array of plain numbers collapsed onto one line.

    Keeps the fixtures reviewable (one record per line-group) without exploding a
    600-element pixel row into 600 lines.
    """
    text = json.dumps(obj, indent=1, sort_keys=False)
    while True:
        collapsed = _FLAT.sub(lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]", text)
        if collapsed == text:
            return collapsed
        text = collapsed


def write(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps_compact(obj))
        handle.write("\n")
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="directory holding inference/*.py at the pinned revision")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "fixtures"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        header = fetch_reference(args.cache, workdir)
        header["generated"] = datetime.date.today().isoformat()
        sys.path.insert(0, workdir)
        install_kernel_stub()
        import image_processor as ip
        import model  # noqa: F401 - imported for get_image_visible / get_window_topk_idxs_visible

        print(f"reference source: {header['source']} @ {REV}")
        crosscheck_load_image(ip, header)
        write(os.path.join(args.out, "geometry.json"), build_geometry_fixture(ip, header))
        write(os.path.join(args.out, "block.json"), build_block_fixture(ip, header))
        write(os.path.join(args.out, "visibility.json"),
              build_visibility_fixture(ip, GEOMETRY["max_n_token"], header))
        write(os.path.join(args.out, "resample.json"), build_resample_fixture(header))


if __name__ == "__main__":
    main()
