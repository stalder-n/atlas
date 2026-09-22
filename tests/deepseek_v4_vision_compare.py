#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Diff Atlas's DeepSeek-V4 aligner rows against DeepSeek's reference tower.

Spark-side bring-up tool for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (and the
`s-zaizen/...-NVFP4` repack). It runs the *reference* `inference/vision.py` on a
CPU, in float32, against the checkpoint's own `vision.*` / `aligner.*` tensors,
and compares the result with the rows Atlas's `DeepseekVisionEncoder` produced
for the same image.

Atlas side of the contract — rank 0 only, `AVAROK_DS_VISION_DUMP=<dir>`:

    <dir>/image_<i>.bf16   raw BF16, row-major [n_llm_h * n_llm_w, hidden]
    <dir>/image_<i>.json   {"n_vit_h":…, "n_vit_w":…, "n_llm_h":…, "n_llm_w":…}

Rows are the *unpermuted* aligner order (row-major over the LLM grid); the
N-layout permutation happens later, at splice time, and is not dumped.

Modes
  --mode trigger   POST one chat/completions with the images, so the server writes the dump
  --mode diff      run the reference and compare it against the dump
  --mode all       trigger, then diff
  --mode selftest  no server, no checkpoint: synthetic weights + positive/negative controls

Typical Spark use:

    python3 tests/deepseek_v4_vision_compare.py --mode all \
        --base-url http://127.0.0.1:8888/v1 --model <served-id> \
        --checkpoint /cache/huggingface/hub/models--deepseek-ai--.../snapshots/<sha> \
        --dump /tmp/ds-vision-dump \
        --image tests/fixtures/images/05_square_768.png \
        --output /tmp/ds-vision-compare.json

The reference sources are fetched from a pinned revision and their sha256 is
verified before they are imported. There is no bypass: this code is executed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REF_REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REF_REVISION = "86f746b36186f0e567729a5c06a8c918caba82a9"
# sha256 of each reference source at REF_REVISION. Verified before import.
REF_FILES = {
    "vision.py": "a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1",
    "image_processor.py": "cac2ff6af15207ce53d0319dc52c6ed1fa5f4fed21f75795fe0fc014632e9086",
}
DEFAULT_REF_CACHE = Path.home() / ".cache" / "atlas" / "dsv4-ref"
# The flat config.json keys the reference `args` object needs.
VISION_KEYS = (
    "vision_n_layers", "vision_dim", "vision_n_heads", "vision_inter_dim",
    "vision_patch_size", "vision_rope_theta", "vision_downsample_ratio",
    "vision_max_n_token", "vision_min_pixels", "vision_max_wh_ratio",
)


class CompareError(RuntimeError):
    """A structural problem: geometry, sizes, missing files, bad reference."""


# ── reference sources ────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_reference(ref_dir: str | None) -> Path:
    """Return a directory holding the two pinned reference sources, verified."""
    if ref_dir:
        directory = Path(ref_dir)
    else:
        directory = DEFAULT_REF_CACHE / REF_REVISION
        directory.mkdir(parents=True, exist_ok=True)
        for name in REF_FILES:
            target = directory / name
            if target.is_file():
                continue
            url = (f"https://huggingface.co/{REF_REPO}/resolve/"
                   f"{REF_REVISION}/inference/{name}")
            print(f"  fetching {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
    for name, want in REF_FILES.items():
        target = directory / name
        if not target.is_file():
            raise CompareError(f"reference source missing: {target}")
        got = _sha256(target)
        if got != want:
            raise CompareError(
                f"{target}: sha256 {got} != pinned {want} for revision "
                f"{REF_REVISION}. Refusing to execute unverified reference code."
            )
    return directory


def import_reference(directory: Path):
    """Import the verified `vision` and `image_processor` modules by path."""
    modules = []
    for name in ("vision", "image_processor"):
        spec = importlib.util.spec_from_file_location(
            f"dsv4_ref_{name}", directory / f"{name}.py")
        if spec is None or spec.loader is None:
            raise CompareError(f"cannot import reference module {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[0], modules[1]


# ── checkpoint ───────────────────────────────────────────────────────────────

def load_config(checkpoint: Path) -> dict:
    path = checkpoint / "config.json"
    if not path.is_file():
        raise CompareError(f"no config.json under {checkpoint}")
    return json.loads(path.read_text())


def reference_args(config: dict) -> SimpleNamespace:
    """Build the reference `args` object from the checkpoint's own config.

    Every value comes from the checkpoint (no literals): a missing vision key is
    a hard error, exactly as Atlas's config parse treats it.
    """
    missing = [k for k in VISION_KEYS if k not in config]
    if missing:
        raise CompareError(f"config.json is not a vision checkpoint; missing {missing}")
    for key in ("hidden_size", "vocab_size"):
        if key not in config:
            raise CompareError(f"config.json missing {key}")
    args = SimpleNamespace(**{k: config[k] for k in VISION_KEYS})
    args.dim = config["hidden_size"]
    args.vocab_size = config["vocab_size"]
    return args


def load_vision_tensors(checkpoint: Path):
    """Load every `vision.*` / `aligner.*` tensor as float32 (BF16 on disk).

    Uses `model.safetensors.index.json` to open only the shard(s) that hold
    them — one ~2 GB shard on both published packings, not the whole 168 GB.
    """
    import torch
    from safetensors import safe_open

    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({v for k, v in weight_map.items()
                         if k.startswith(("vision.", "aligner."))})
        if not shards:
            raise CompareError(f"{index}: no vision./aligner. tensors in the weight map")
    else:
        shards = sorted(p.name for p in checkpoint.glob("*.safetensors"))
        if not shards:
            raise CompareError(f"no safetensors shards under {checkpoint}")
    tensors: dict = {}
    for shard in shards:
        with safe_open(str(checkpoint / shard), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(("vision.", "aligner.")):
                    tensors[key] = handle.get_tensor(key).to(torch.float32)
    if not tensors:
        raise CompareError(f"{checkpoint}: no vision./aligner. tensors found")
    return tensors


def build_reference_towers(vision_mod, args, tensors):
    """Instantiate the reference ViT + Aligner and load the checkpoint weights."""
    vit = vision_mod.ViT(args)
    aligner = vision_mod.Aligner(args)
    for module, prefix in ((vit, "vision."), (aligner, "aligner.")):
        state = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise CompareError(
                f"{prefix}* does not match the reference module: "
                f"missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}")
    return vit.eval(), aligner.eval()


def reference_encode(image_path: str, args, image_processor_mod, vit, aligner):
    """Run the reference tower on one image. Returns (rows_f32, geometry dict)."""
    import numpy as np
    import torch

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor_mod.load_image(
        {"url": image_path}, args)
    with torch.no_grad():
        hidden = vit(patches.to(torch.float32), n_vit_h, n_vit_w)
        rows = aligner(hidden, n_vit_h, n_vit_w)
    array = rows.to(torch.float32).numpy().astype(np.float32, copy=True)
    geometry = {"n_vit_h": n_vit_h, "n_vit_w": n_vit_w,
                "n_llm_h": n_llm_h, "n_llm_w": n_llm_w}
    return array, geometry


# ── Atlas dump ───────────────────────────────────────────────────────────────

def bf16_to_f32(raw: bytes):
    import numpy as np
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).copy()


def read_dump(dump_dir: Path, index: int, hidden: int):
    """Read `image_<i>.bf16` + `image_<i>.json`. Returns (rows_f32, geometry)."""
    import numpy as np

    meta_path = dump_dir / f"image_{index}.json"
    rows_path = dump_dir / f"image_{index}.bf16"
    if not rows_path.is_file():
        raise CompareError(
            f"{rows_path} missing. AVAROK_DS_VISION_DUMP produced no rows for image "
            f"{index} — is it set on rank 0 (the only rank that owns the ViT), and is "
            "the directory mounted out of the container?")
    if not meta_path.is_file():
        raise CompareError(f"{meta_path} missing: the dump has rows but no geometry sidecar")
    geometry = json.loads(meta_path.read_text())
    for key in ("n_vit_h", "n_vit_w", "n_llm_h", "n_llm_w"):
        if not isinstance(geometry.get(key), int):
            raise CompareError(f"{meta_path}: missing or non-integer {key}")
    expect = geometry["n_llm_h"] * geometry["n_llm_w"] * hidden * 2
    actual = rows_path.stat().st_size
    if actual != expect:
        raise CompareError(
            f"{rows_path}: {actual} bytes, expected {expect} "
            f"({geometry['n_llm_h']}x{geometry['n_llm_w']} rows x {hidden} x 2 bytes BF16)")
    rows = bf16_to_f32(rows_path.read_bytes())
    return rows.reshape(geometry["n_llm_h"] * geometry["n_llm_w"], hidden), geometry


def write_dump(dump_dir: Path, index: int, rows, geometry: dict) -> None:
    """Write a dump in Atlas's contract format (used by --mode selftest)."""
    import numpy as np
    dump_dir.mkdir(parents=True, exist_ok=True)
    bits = (rows.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    (dump_dir / f"image_{index}.bf16").write_bytes(bits.tobytes())
    (dump_dir / f"image_{index}.json").write_text(json.dumps(geometry) + "\n")


# ── metrics ──────────────────────────────────────────────────────────────────

def compare_rows(atlas, reference) -> dict:
    import numpy as np
    a = atlas.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    diff = a - b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
    rms = float(np.sqrt(np.mean(diff ** 2)))
    ref_rms = float(np.sqrt(np.mean(b ** 2)))
    per_row = np.abs(atlas.astype(np.float64) - reference.astype(np.float64)).max(axis=1)
    worst = np.argsort(per_row)[::-1][:5]
    return {
        "rows": int(atlas.shape[0]), "hidden": int(atlas.shape[1]),
        "cosine": cosine, "max_abs": float(np.max(np.abs(diff))), "rms": rms,
        "relative_rms": float(rms / ref_rms) if ref_rms > 0 else float("inf"),
        "worst_rows": [{"row": int(r), "max_abs": float(per_row[r])} for r in worst],
    }


# ── modes ────────────────────────────────────────────────────────────────────

def image_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    payload = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{payload}"


def mode_trigger(args) -> None:
    """One request carrying every --image, so rank 0 writes the dump."""
    if not args.model:
        raise CompareError("--mode trigger needs --model")
    content = [{"type": "image_url", "image_url": {"url": image_data_uri(p)}}
               for p in args.image]
    content.append({"type": "text", "text": "Describe each image in one sentence."})
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 16, "temperature": 0, "stream": False}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("ATLAS_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.load(response)
    usage = payload.get("usage", {})
    print(f"  triggered: prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}", flush=True)


def mode_diff(args) -> int:
    reference_dir = resolve_reference(args.ref_dir)
    vision_mod, image_processor_mod = import_reference(reference_dir)
    checkpoint = Path(args.checkpoint)
    config = load_config(checkpoint)
    ref_args = reference_args(config)
    hidden = ref_args.dim

    tensors = load_vision_tensors(checkpoint)
    vit, aligner = build_reference_towers(vision_mod, ref_args, tensors)

    dump_dir = Path(args.dump)
    if not dump_dir.is_dir():
        raise CompareError(
            f"{dump_dir} does not exist. AVAROK_DS_VISION_DUMP produced no rows — is it "
            "set on rank 0 (the only rank that owns the ViT), and is the directory "
            "mounted out of the container?")

    report = {"revision": REF_REVISION, "checkpoint": str(checkpoint),
              "min_cos": args.min_cos, "images": []}
    failures = 0
    for index, image_path in enumerate(args.image):
        atlas_rows, atlas_geometry = read_dump(dump_dir, index, hidden)
        ref_rows, ref_geometry = reference_encode(
            image_path, ref_args, image_processor_mod, vit, aligner)
        # Geometry first: a grid mismatch makes every numeric metric meaningless.
        if atlas_geometry != ref_geometry:
            raise CompareError(
                f"image {index} ({image_path}): geometry mismatch — Atlas dumped "
                f"{atlas_geometry}, the reference computed {ref_geometry}. Fix the "
                "resampler/grid solver before reading any numbers.")
        if atlas_rows.shape != ref_rows.shape:
            raise CompareError(
                f"image {index}: shape {atlas_rows.shape} != reference {ref_rows.shape}")
        metrics = compare_rows(atlas_rows, ref_rows)
        metrics.update({"index": index, "image": image_path, "geometry": ref_geometry})
        metrics["pass"] = metrics["cosine"] >= args.min_cos
        failures += 0 if metrics["pass"] else 1
        report["images"].append(metrics)
        print(f"  image {index}  {Path(image_path).name:28s} "
              f"{ref_geometry['n_llm_h']}x{ref_geometry['n_llm_w']} rows "
              f"cos={metrics['cosine']:.6f} max|d|={metrics['max_abs']:.4g} "
              f"rel_rms={metrics['relative_rms']:.4g} "
              f"{'PASS' if metrics['pass'] else 'FAIL'}", flush=True)
        for row in metrics["worst_rows"]:
            print(f"      worst row {row['row']:5d}  max|d|={row['max_abs']:.4g}")
    report["pass"] = failures == 0
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"  report written to {args.output}")
    print("PASS" if report["pass"] else f"FAIL ({failures} image(s) below cosine {args.min_cos})")
    return 0 if report["pass"] else 1


# ── selftest ─────────────────────────────────────────────────────────────────

SELFTEST_CONFIG = {
    "model_type": "deepseek_v4", "hidden_size": 64, "vocab_size": 129280,
    "vision_n_layers": 2, "vision_dim": 32, "vision_n_heads": 4,
    "vision_inter_dim": 48, "vision_patch_size": 14, "vision_rope_theta": 10000.0,
    "vision_downsample_ratio": 3, "vision_max_n_token": 384,
    "vision_min_pixels": 147456, "vision_max_wh_ratio": 8,
}


def _selftest_checkpoint(root: Path, vision_mod, ref_args) -> Path:
    """A tiny but structurally real checkpoint: seeded weights, same key names."""
    import torch
    from safetensors.torch import save_file

    checkpoint = root / "ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps(SELFTEST_CONFIG))
    torch.manual_seed(20260921)
    tensors = {}
    for module, prefix in ((vision_mod.ViT(ref_args), "vision."),
                           (vision_mod.Aligner(ref_args), "aligner.")):
        for name, value in module.state_dict().items():
            tensors[prefix + name] = (torch.randn_like(value) * 0.05).to(torch.bfloat16)
    save_file(tensors, str(checkpoint / "model-00001-of-00001.safetensors"))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors"
                                        for k in tensors}}))
    return checkpoint


def _selftest_image(path: Path) -> None:
    """504x504 — the size the reference solver passes through untouched."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(20260921)
    pixels = rng.integers(0, 256, size=(504, 504, 3), dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)


def _selftest_args(checkpoint: Path, dump: Path, image: Path,
                   reference_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint), dump=str(dump), image=[str(image)],
        ref_dir=str(reference_dir), min_cos=0.999, output=None)


def mode_selftest(args) -> int:
    """Host-only proof that the comparison actually discriminates.

    Positive control: a dump written *from* the reference must match it (only the
    BF16 round-trip differs). Negative controls: a perturbed row, a truncated
    row file, and a wrong grid must each be caught.
    """
    import numpy as np

    reference_dir = resolve_reference(args.ref_dir)
    vision_mod, image_processor_mod = import_reference(reference_dir)
    ref_args = reference_args(SELFTEST_CONFIG)

    failures = []
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        checkpoint = _selftest_checkpoint(root, vision_mod, ref_args)
        image = root / "synthetic_504.png"
        _selftest_image(image)
        tensors = load_vision_tensors(checkpoint)
        vit, aligner = build_reference_towers(vision_mod, ref_args, tensors)
        rows, geometry = reference_encode(
            str(image), ref_args, image_processor_mod, vit, aligner)
        print(f"  reference geometry {geometry}, rows {rows.shape}", flush=True)
        if geometry != {"n_vit_h": 36, "n_vit_w": 36, "n_llm_h": 12, "n_llm_w": 12}:
            failures.append(f"504x504 must pass through unresized, got {geometry}")

        dump = root / "dump"
        write_dump(dump, 0, rows, geometry)

        # Positive control.
        code = mode_diff(_selftest_args(checkpoint, dump, image, reference_dir))
        if code != 0:
            failures.append("positive control: a BF16 round-trip of the reference failed")

        # Negative control 1 — one perturbed row.
        perturbed = rows.copy()
        perturbed[7] += np.float32(50.0)
        write_dump(dump, 0, perturbed, geometry)
        if mode_diff(_selftest_args(checkpoint, dump, image, reference_dir)) == 0:
            failures.append("negative control: a perturbed row was not caught")

        # Negative control 2 — truncated row file.
        write_dump(dump, 0, rows, geometry)
        path = dump / "image_0.bf16"
        path.write_bytes(path.read_bytes()[:-2])
        try:
            mode_diff(_selftest_args(checkpoint, dump, image, reference_dir))
            failures.append("negative control: a truncated dump was not caught")
        except CompareError as error:
            print(f"  caught truncation: {error}", flush=True)

        # Negative control 3 — a wrong grid in the sidecar that keeps the byte
        # count valid, so only the geometry cross-check can catch it.
        write_dump(dump, 0, rows, dict(geometry, n_vit_w=geometry["n_vit_w"] + 1))
        try:
            mode_diff(_selftest_args(checkpoint, dump, image, reference_dir))
            failures.append("negative control: a wrong grid was not caught")
        except CompareError as error:
            print(f"  caught geometry mismatch: {error}", flush=True)

    for failure in failures:
        print(f"  [FAIL] {failure}")
    print("SELFTEST PASS" if not failures else f"SELFTEST FAIL ({len(failures)})")
    return 0 if not failures else 1


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["trigger", "diff", "all", "selftest"],
                        default="all")
    parser.add_argument("--base-url", default="http://localhost:8888/v1")
    parser.add_argument("--model", help="served model id (trigger)")
    parser.add_argument("--checkpoint", help="checkpoint snapshot directory (diff)")
    parser.add_argument("--dump", default="/tmp/ds-vision-dump",
                        help="the directory AVAROK_DS_VISION_DUMP wrote (rank 0)")
    parser.add_argument("--image", action="append", default=[],
                        help="image path, repeatable; order must match image_<i>")
    parser.add_argument("--ref-dir", help="directory holding the pinned reference sources")
    parser.add_argument("--min-cos", type=float, default=0.99,
                        help="bring-up signal, not a certified tolerance")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args()

    try:
        if args.mode == "selftest":
            return mode_selftest(args)
        if not args.image:
            raise CompareError("--image is required (repeat it, in dump order)")
        if args.mode in ("trigger", "all"):
            mode_trigger(args)
        if args.mode == "trigger":
            return 0
        if not args.checkpoint:
            raise CompareError("--checkpoint is required for --mode diff/all")
        return mode_diff(args)
    except CompareError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
