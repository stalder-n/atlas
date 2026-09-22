#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""DeepSeek-V4 vision PROMPT golden generator (unit U2.4).

Every expected prompt string in `deepseek_v4_vision.rs` is produced by DeepSeek's OWN
`encoding/encoding_dsv4.py` at a pinned revision. Nothing is hand-written: this script
fetches the reference encoder from Hugging Face (falling back to a local cache), feeds it
the EXACT JSON message array Atlas's server lowering emits, and prints the resulting
prompts as Rust `const` literals.

    # regenerate (prints the block to stdout)
    /home/coder/.venvs/dsv4/bin/python3 \
        crates/spark-server/src/tokenizer/tests/gen_deepseek_v4_vision_goldens.py

    # verify the committed literals still match the reference
    /home/coder/.venvs/dsv4/bin/python3 \
        crates/spark-server/src/tokenizer/tests/gen_deepseek_v4_vision_goldens.py \
        --check crates/spark-server/src/tokenizer/tests/deepseek_v4_vision.rs

`encoding_dsv4.py` is pure stdlib, so this script also runs under a plain `python3` on the
Spark; the venv path above is only what this GPU-less host has.

# Two deliberate differences between the generator input and the Atlas input

1. Atlas's `build_json_messages_ds` emits a BARE `{"type": "image"}` marker per media item
   (the image bytes travel beside the JSON, in the ordered `Vec<VisionItem>`). The
   reference's `_extract_image` raises on a block with no `source`/`url`/`data`, so this
   script swaps each bare marker for `{"type":"image_url","image_url":{"url": ...}}` before
   calling the reference. The swap cannot change the emitted PROMPT: `_process_image_blocks`
   replaces every image block with the same `{"type":"text","text": PLACEHOLDER}` regardless
   of which of the two spellings it matched, and the extracted record goes to the separate
   `media_data["images"]` channel that Atlas does not port. Each case asserts both the
   placeholder count and that record count below.

2. `tool_calls[*].function.arguments` is a JSON **object** in the Atlas message array (the IR
   parses it at the adapter boundary; `normalize_tool_call_arguments`/F76 does the same for a
   stringified one), while the reference's `encode_arguments_to_dsml` calls `json.loads` on it
   and needs the OpenAI wire **string**. This script re-serialises each `arguments` object
   before handing the case to the reference. Passing the object through instead silently takes
   the reference's `except` branch and renders one bogus `name="arguments"` parameter.

3. The message arrays here are the ATLAS-NORMALISED ones, not the client's. `MsgEntry` keeps
   `content: String` plus `media: Vec<MediaKind>`, so an interleaved `[text, image, text]`
   turn is lowered as `[image, text("beforeafter")]` — images first, then the concatenated
   text. Case G2 is generated from that normalised form on purpose, and the divergence from
   a literal reading of the reference is an ACCEPTED DEVIATION recorded for the bring-up
   doc, not something these goldens paper over.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import sys
import tempfile
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REV = "86f746b36186f0e567729a5c06a8c918caba82a9"
DEFAULT_CACHE = (
    "/tmp/claude-1000/-home-coder-dev-atlas/"
    "f9e8e1ea-da7e-4397-9cb7-8d96789896da/scratchpad/dsv4"
)
REF_FILE = "encoding_dsv4.py"

BEGIN = "// ---- BEGIN GENERATED GOLDENS (gen_deepseek_v4_vision_goldens.py) ----"
END = "// ---- END GENERATED GOLDENS ----"

# Inert: the URL only lands in the reference's `media_data["images"]` channel, never in the
# prompt. Atlas carries the real bytes in `PreparedChat.image_pixels` instead.
IMG = {"type": "image"}


def img_text(text):
    return {"type": "text", "text": text}


# The Atlas-normalised JSON message arrays, exactly as
# `api::chat::template::build_json_messages_ds` emits them.
CASES = [
    {
        "name": "G1",
        "doc": "single image then text, chat mode (OpenAI `image_url` part)",
        "thinking": "chat",
        "effort": None,
        "tools": None,
        "messages": [{"role": "user", "content": [IMG, img_text("describe")]}],
    },
    {
        "name": "G2",
        "doc": (
            "interleaved text/image/text, in the ATLAS-NORMALISED form the IR can "
            "represent (image first, the two text parts concatenated) — accepted deviation"
        ),
        "thinking": "chat",
        "effort": None,
        "tools": None,
        "messages": [{"role": "user", "content": [IMG, img_text("beforeafter")]}],
    },
    {
        "name": "G3",
        "doc": "two images in one turn, then text",
        "thinking": "chat",
        "effort": None,
        "tools": None,
        "messages": [{"role": "user", "content": [IMG, IMG, img_text("compare")]}],
    },
    {
        "name": "G4",
        "doc": "image inside a tool result (the reference's own tool-role image case)",
        "thinking": "chat",
        "effort": None,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "take a screenshot",
                    "parameters": {
                        "type": "object",
                        "properties": {"region": {"type": "string"}},
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "grab a shot"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "screenshot",
                            "arguments": {"region": "full"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": [IMG, img_text("tool image")],
                "tool_call_id": "call-1",
            },
        ],
    },
    {
        "name": "G5",
        "doc": "Anthropic `image` block — `to_ir` lowers it to the same images-then-text IR",
        "thinking": "chat",
        "effort": None,
        "tools": None,
        "messages": [{"role": "user", "content": [IMG, img_text("anthropic look")]}],
    },
    {
        "name": "G6",
        "doc": "image on an ASSISTANT turn, thinking mode with `low` effort",
        "thinking": "thinking",
        "effort": "low",
        "tools": None,
        "messages": [
            {"role": "user", "content": "first look"},
            {"role": "assistant", "content": [IMG, img_text("I see a cat")]},
            {"role": "user", "content": "and now?"},
        ],
    },
]


def fetch_reference(cache_dir, workdir):
    """HF first, local cache second. Returns the header dict recorded in the output."""
    info = {"repo": REPO, "revision": REV, "source": "hf", "files": {}}
    url = f"https://huggingface.co/{REPO}/raw/{REV}/encoding/{REF_FILE}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
    except Exception as exc:  # noqa: BLE001 - any network/HTTP failure falls back
        print(f"  ! HF fetch failed ({exc}); using cache", file=sys.stderr)
        info["source"] = "cache"
        with open(os.path.join(cache_dir, "encoding", REF_FILE), "rb") as handle:
            body = handle.read()
    path = os.path.join(workdir, REF_FILE)
    with open(path, "wb") as handle:
        handle.write(body)
    info["files"][REF_FILE] = hashlib.sha256(body).hexdigest()
    return info, path


def load_module(path):
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_reference_messages(messages, url):
    """Atlas's message array → the shape the reference encoder expects.

    Two rewrites, both documented in the module docstring: bare image markers gain a
    source, and parsed `arguments` objects go back to their OpenAI wire strings.
    """
    out = json.loads(json.dumps(messages))
    count = 0
    for msg in out:
        for call in msg.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments")
            if not isinstance(arguments, str):
                call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image" and "url" not in block:
                content[i] = {"type": "image_url", "image_url": {"url": url}}
                count += 1
    return out, count


def render_case(enc, case):
    messages, n_images = to_reference_messages(case["messages"], "data:image/png;base64,AA==")
    if case["tools"]:
        messages[0]["tools"] = case["tools"]
    prompt, media = enc.encode_messages(
        messages,
        thinking_mode=case["thinking"],
        reasoning_effort=case["effort"],
        return_multi_modal_data=True,
    )
    got = prompt.count(enc.IMAGE_PLACEHOLDER)
    if got != n_images:
        raise SystemExit(f"{case['name']}: {got} placeholders for {n_images} images")
    if len(media["images"]) != n_images:
        raise SystemExit(f"{case['name']}: reference collected {len(media['images'])} records")
    return prompt, n_images


def rust_str(text):
    out = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append("\\u{%x}" % ord(char))
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def build_block(enc, header, invocation):
    lines = [BEGIN]
    lines.append("// Produced by DeepSeek's own `encoding/encoding_dsv4.py`, NOT hand-written.")
    lines.append(f"//   repo:     {header['repo']}")
    lines.append(f"//   revision: {header['revision']}")
    lines.append(f"//   source:   {header['source']}")
    lines.append(f"//   sha256:   {REF_FILE} = {header['files'][REF_FILE]}")
    lines.append(f"//   generated:{header['generated']}")
    lines.append(f"//   invoked:  {invocation}")
    lines.append(f"// Placeholder: {enc.IMAGE_PLACEHOLDER}")
    lines.append("/// The reference's `IMAGE_PLACEHOLDER`, added-token id 129264.")
    lines.append(f"pub(super) const PLACEHOLDER: &str = {rust_str(enc.IMAGE_PLACEHOLDER)};")
    for case in CASES:
        prompt, n_images = render_case(enc, case)
        lines.append(
            f"/// {case['name']} [{case['thinking']}, effort={case['effort']}, "
            f"images={n_images}, tools={'yes' if case['tools'] else 'no'}] {case['doc']}"
        )
        lines.append(f"const {case['name']}: &str = {rust_str(prompt)};")
    lines.append(END)
    return "\n".join(lines) + "\n"


LITERAL_RE = re.compile(r"const\s+(\w+)\s*:\s*&str\s*=\s*(\"(?:[^\"\\\\]|\\\\.)*\")\s*;")


def literals(text):
    """name -> Rust string literal, insensitive to how rustfmt wrapped the statement.

    The header lines carry a generation date and an invocation that legitimately differ
    between runs, and rustfmt moves a long literal onto its own line; only the literals
    themselves are compared.
    """
    return {m.group(1): m.group(2) for m in LITERAL_RE.finditer(text)}


def extract_block(path):
    with open(path, encoding="utf-8") as handle:
        body = handle.read()
    start = body.find(BEGIN)
    stop = body.find(END)
    if start < 0 or stop < 0:
        raise SystemExit(f"{path}: generated-goldens markers not found")
    return body[start : stop + len(END)] + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="local reference cache dir")
    parser.add_argument("--check", metavar="PATH", help="diff against a committed test file")
    args = parser.parse_args()

    invocation = " ".join(shlex.quote(a) for a in [os.path.basename(sys.argv[0])] + sys.argv[1:])
    with tempfile.TemporaryDirectory() as workdir:
        header, path = fetch_reference(args.cache, workdir)
        header["generated"] = datetime.date.today().isoformat()
        enc = load_module(path)
        block = build_block(enc, header, invocation)

    if not args.check:
        sys.stdout.write(block)
        return

    committed = extract_block(args.check)
    want, have = literals(block), literals(committed)
    if want == have:
        print(f"OK: {args.check} matches the reference at {REV}")
        return
    for name in sorted(set(want) | set(have)):
        if want.get(name) != have.get(name):
            print(f"MISMATCH {name}\n  reference: {want.get(name)}\n  committed: {have.get(name)}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
