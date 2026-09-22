#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Regenerate `fixtures/vision-exp-tensor-headers.json`.

Reads the safetensors HEADERS (dtype + shape + data_offsets) of the tensors the
native-DSpark preflight contract validates, for BOTH Vision-Exp packings, at
pinned revisions. No tensor payload is ever downloaded: the index names the
shard, an 8-byte range read gives the header length, and a second range read
gives the header JSON.

Usage (from the repository root):

    /home/coder/.venvs/dsv4/bin/python3 \\
        crates/spark-model/src/preflight/deepseek_v4/gen_tensor_header_fixture.py

The script itself needs only the standard library, so it also runs under a
plain `python3` on the Spark. `--cache-dir` points at a local snapshot to use
instead of the network (a directory holding `model.safetensors.index.json` and
the shards, per packing key).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from collections import defaultdict

HF = "https://huggingface.co"

# (key, repo id, pinned revision). Both are MIT; the revisions are the ones
# `docs/DEEPSEEK_V4_VISION_EXP_BRINGUP.md` §1 pins.
PACKINGS = [
    (
        "official",
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "6821d6ad3681a4b137b066b76094fa82ebd0a380",
    ),
    (
        "nvfp4",
        "s-zaizen/DeepSeek-V4-Flash-Vision-Exp-NVFP4",
        "325721df897916a21f71ba288b1c54f8de12d966",
    ),
]

# Routed experts are 256 per stage; two are enough to pin the packing (EP rank 0
# and rank 1 under EP=2) and keep the fixture a few hundred entries instead of
# ~9k.
SAMPLE_EXPERTS = (0, 128)

EXPERT_RE = re.compile(r"\.ffn\.experts\.(\d+)\.")


def wanted(name: str) -> bool:
    """Every `mtp.*` tensor the DSpark contract reads, plus one target layer.

    The target layer is carried so a reader can diff the stage bodies against a
    real V4 layer of the same checkpoint without opening a second fixture.
    """
    if name.startswith("mtp."):
        m = EXPERT_RE.search(name)
        return m is None or int(m.group(1)) in SAMPLE_EXPERTS
    if name.startswith("layers.42."):
        m = EXPERT_RE.search(name)
        return m is None or int(m.group(1)) in SAMPLE_EXPERTS
    return False


def fetch(url: str, rng: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url)
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def shard_header(repo: str, rev: str, shard: str, cache_dir: str | None) -> dict:
    """The safetensors header of one shard: `{name: {dtype, shape, data_offsets}}`."""
    if cache_dir:
        path = os.path.join(cache_dir, shard)
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            return json.loads(fh.read(n))
    url = f"{HF}/{repo}/resolve/{rev}/{shard}"
    (n,) = struct.unpack("<Q", fetch(url, (0, 7)))
    return json.loads(fetch(url, (8, 8 + n - 1)))


def index(repo: str, rev: str, cache_dir: str | None) -> dict:
    if cache_dir:
        with open(os.path.join(cache_dir, "model.safetensors.index.json")) as fh:
            return json.load(fh)
    return json.loads(fetch(f"{HF}/{repo}/resolve/{rev}/model.safetensors.index.json"))


def collect(repo: str, rev: str, cache_dir: str | None) -> dict:
    weight_map = index(repo, rev, cache_dir)["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if wanted(name):
            by_shard[shard].append(name)

    out: dict[str, dict] = {}
    for shard in sorted(by_shard):
        header = shard_header(repo, rev, shard, cache_dir)
        for name in sorted(by_shard[shard]):
            entry = header[name]
            out[name] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": entry["data_offsets"],
            }
        print(f"  {shard}: {len(by_shard[shard])} headers", file=sys.stderr)
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.join(here, "fixtures", "vision-exp-tensor-headers.json")
    )
    ap.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        metavar="KEY=DIR",
        help="local snapshot for one packing key (repeatable)",
    )
    args = ap.parse_args()
    caches = dict(spec.split("=", 1) for spec in args.cache_dir)

    doc: dict[str, object] = {
        "_generator": {
            "script": "crates/spark-model/src/preflight/deepseek_v4/gen_tensor_header_fixture.py",
            "invocation": (
                "/home/coder/.venvs/dsv4/bin/python3 "
                "crates/spark-model/src/preflight/deepseek_v4/gen_tensor_header_fixture.py"
            ),
            "what": (
                "safetensors HEADERS ONLY (dtype, shape, data_offsets) for every mtp.* "
                "tensor of the native 3-stage DSpark module plus target layer 42, "
                "routed experts sampled at %s. No tensor payload." % (list(SAMPLE_EXPERTS),)
            ),
        }
    }
    for key, repo, rev in PACKINGS:
        print(f"{key}: {repo}@{rev}", file=sys.stderr)
        doc[key] = {
            "_source": {"repo": repo, "revision": rev, "license": "MIT"},
            "tensors": collect(repo, rev, caches.get(key)),
        }

    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
