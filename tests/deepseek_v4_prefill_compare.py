# SPDX-License-Identifier: AGPL-3.0-only
"""Collect/compare whole-prefill FP32 logits on EP2 Sparks, standard library only.

Uses the existing ATLAS_NEMO_DUMP/atlas_logits.bin diagnostic. Run only with
exclusive access to rank 0 and its dump directory. No timings are collected.
"""
import argparse
import array
import datetime
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

BOUNDARIES = (3, 4, 5, 15, 16, 17, 31, 32, 33, 127, 128, 129,
              255, 256, 257, 1024, 1536, 2047)
MODES = ('baseline', 'fp8', 'tc', 'both')
# These are operator-attested inputs; the HTTP API cannot attest container env.
FINGERPRINT_KEYS = ('source_sha', 'image_id', 'checkpoint', 'checkpoint_revision',
                    'rank0_host', 'rank1_host', 'rank0_serve_args', 'rank1_serve_args',
                    'common_environment', 'recipe_sha', 'vocab_size')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_logits(reference, candidate, atol, rtol):
    require(all(math.isfinite(t) and t >= 0 for t in (atol, rtol)),
            'tolerances must be finite and nonnegative')
    require(len(reference) > 0 and len(reference) == len(candidate),
            'missing logits or different vocabulary sizes')
    require(all(math.isfinite(x) for row in (reference, candidate) for x in row),
            'nonfinite raw logits')
    errors = [abs(a - b) for a, b in zip(reference, candidate)]
    violations = sum(e > atol + rtol * abs(a) for a, e in zip(reference, errors))
    ref_argmax = max(range(len(reference)), key=reference.__getitem__)
    got_argmax = max(range(len(candidate)), key=candidate.__getitem__)
    return {'pass': violations == 0, 'violations': violations,
            'max_abs_error': max(errors),
            'rms_error': math.sqrt(sum(e * e for e in errors) / len(errors)),
            'reference_argmax': ref_argmax, 'candidate_argmax': got_argmax,
            'argmax_equal': ref_argmax == got_argmax}


def validate_pair(reference, candidate):
    require(reference['complete'] and candidate['complete'], 'incomplete capture')
    require(reference['fingerprint'] == candidate['fingerprint'],
            'source/image/checkpoint/serve fingerprints differ')
    require(reference['requests'] == candidate['requests'], 'request settings differ')
    require(reference['mode'] == 'baseline', 'reference must be baseline mode')
    require(candidate['mode'] in MODES, 'unknown candidate mode')


def read_logits(path, vocab_size):
    raw = path.read_bytes()
    require(len(raw) == vocab_size * 4, f'{path}: expected exactly one full FP32 row')
    values = array.array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    require(all(math.isfinite(v) for v in values), f'{path}: nonfinite logits')
    return values


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def post(base_url, endpoint, body, timeout):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('ATLAS_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['ATLAS_API_KEY']
    request = urllib.request.Request(base_url.rstrip('/') + endpoint,
                                     data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def collect(args):
    fingerprint = json.loads(Path(args.fingerprint).read_text())
    for key in FINGERPRINT_KEYS:
        require(key in fingerprint and fingerprint[key] not in (None, '', [], {}),
                f'fingerprint missing {key}')
    vocab = fingerprint['vocab_size']
    require(isinstance(vocab, int) and vocab > 0, 'invalid vocabulary size')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    # Same code-like text and exact token IDs for all four modes. The completions
    # endpoint feeds these IDs verbatim, avoiding chat-template length ambiguity.
    text = ('def prefix_sum(values):\n    total = 0\n    for value in values:\n'
            '        total += value\n        yield total\n\n') * 512
    tokenized = post(args.base_url, '/tokenize', {'model': args.model, 'prompt': text},
                     args.timeout)
    tokens = tokenized['tokens']
    require(len(tokens) >= max(BOUNDARIES), 'tokenization fixture too short')
    requests = [dict(model=args.model, prompt=tokens[:n], max_tokens=1,
                     temperature=0, top_k=0, top_p=1, min_p=0,
                     repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                     seed=42, stream=False) for n in BOUNDARIES]
    manifest = {'version': 1, 'fingerprint': fingerprint, 'mode': args.mode,
                'base_url': args.base_url,
                'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'requests': requests, 'complete': False}
    save(out / 'manifest.json', manifest)
    dump = Path(args.dump_dir) / 'atlas_logits.bin'
    require(dump.parent.is_dir(), 'rank-0 dump mount does not exist')
    for n, body in zip(BOUNDARIES, requests):
        # Removing the previous diagnostic prevents a silently failed dump from
        # being mistaken for the current request. No other files are removed.
        dump.unlink(missing_ok=True)
        save(out / f'{n}.request.json', body)
        response = post(args.base_url, '/v1/completions', body, args.timeout)
        save(out / f'{n}.response.json', response)
        require(response['usage']['prompt_tokens'] == n, f'{n}: prompt length changed')
        require(response['usage']['completion_tokens'] == 1, f'{n}: missing completion')
        require(dump.is_file(), f'{n}: prefill dump missing; verify ATLAS_NEMO_DUMP mount')
        read_logits(dump, vocab)
        (out / f'{n}.f32').write_bytes(dump.read_bytes())
        print(f'captured {n} input tokens', flush=True)
    manifest['complete'] = True
    save(out / 'manifest.json', manifest)


def compare(args):
    reference, candidate = Path(args.reference), Path(args.candidate)
    ref_manifest = json.loads((reference / 'manifest.json').read_text())
    got_manifest = json.loads((candidate / 'manifest.json').read_text())
    validate_pair(ref_manifest, got_manifest)
    vocab = ref_manifest['fingerprint']['vocab_size']
    results = {}
    for n in BOUNDARIES:
        results[str(n)] = compare_logits(read_logits(reference / f'{n}.f32', vocab),
                                         read_logits(candidate / f'{n}.f32', vocab),
                                         args.atol, args.rtol)
    passed = all(r['pass'] and (r['argmax_equal'] or not args.require_argmax)
                 for r in results.values())
    report = {'pass': passed, 'atol': args.atol, 'rtol': args.rtol,
              'require_argmax': args.require_argmax, 'results': results,
              'reference': str(reference), 'candidate': str(candidate),
              'fingerprint': ref_manifest['fingerprint']}
    save(Path(args.output), report)
    print('PASS' if passed else 'FAIL')
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('collect')
    for option in ('base-url', 'model', 'fingerprint', 'dump-dir', 'output'):
        capture.add_argument('--' + option, required=True)
    capture.add_argument('--mode', required=True, choices=MODES)
    capture.add_argument('--timeout', type=float, default=1800)
    check = commands.add_parser('compare')
    for option in ('reference', 'candidate', 'output'):
        check.add_argument('--' + option, required=True)
    check.add_argument('--atol', type=float, required=True)
    check.add_argument('--rtol', type=float, required=True)
    check.add_argument('--require-argmax', action='store_true')
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    sys.exit(main())
