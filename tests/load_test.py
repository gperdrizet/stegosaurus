'''Load test for the Stegosaurus Gradio app.

Submits concurrent encode and decode requests against a running instance
(local or Cloud Run) and reports latency statistics.

Usage
-----
# Against a local instance:
python tests/load_test.py --url http://localhost:8080

# Against Cloud Run:
python tests/load_test.py --url https://<your-service>.run.app --workers 8

# Full options:
python tests/load_test.py --help
'''

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gradio_client import Client

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

_PROMPT = 'A turtle and a bird were walking in the forest one day. The turtle said, "'

# Short messages keep encode time predictable; mix lengths for realism
_ENCODE_MESSAGES = [
    'hello',
    'secret',
    'load test',
    'stegosaurus',
    'hello world',
]

# Pre-encoded cover texts that correspond to _PROMPT + message above.
# If empty, the test skips decode jobs and runs encode-only.
# Populate by running encode once and pasting the outputs here.
_DECODE_COVER_TEXTS: list[str] = []

# ---------------------------------------------------------------------------
# Single-request helpers
# ---------------------------------------------------------------------------

def _encode_one(client: Client, message: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        # Gradio SSE: first .then() output is [cover_text, status]
        result = client.predict(
            prompt,
            message,
            api_name='/encode_message',
        )
        elapsed = time.perf_counter() - t0
        cover_text, status = result
        ok = not status or 'Error' not in status
        return {'kind': 'encode', 'ok': ok, 'elapsed': elapsed,
                'error': status if not ok else None, 'cover_text': cover_text}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {'kind': 'encode', 'ok': False, 'elapsed': elapsed, 'error': str(exc)}


def _decode_one(client: Client, cover_text: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        result = client.predict(
            prompt,
            cover_text,
            api_name='/decode_message',
        )
        elapsed = time.perf_counter() - t0
        message, status = result
        ok = not status or 'Error' not in status
        return {'kind': 'decode', 'ok': ok, 'elapsed': elapsed,
                'error': status if not ok else None}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {'kind': 'decode', 'ok': False, 'elapsed': elapsed, 'error': str(exc)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    url: str,
    total_requests: int,
    workers: int,
    prompt: str,
) -> list[dict]:
    '''Submit total_requests jobs using a thread pool of size workers.'''

    have_decode = bool(_DECODE_COVER_TEXTS)
    jobs = []

    for i in range(total_requests):
        if not have_decode or i % 2 == 0:
            msg = _ENCODE_MESSAGES[i % len(_ENCODE_MESSAGES)]
            jobs.append(('encode', msg))
        else:
            cover = _DECODE_COVER_TEXTS[(i // 2) % len(_DECODE_COVER_TEXTS)]
            jobs.append(('decode', cover))

    results = []

    print(f'Submitting {len(jobs)} requests with {workers} concurrent workers…')
    print(f'Target: {url}\n')

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Each thread gets its own Client instance (not thread-safe to share)
        futures = {}
        for kind, payload in jobs:
            client = Client(url, verbose=False)
            if kind == 'encode':
                fut = pool.submit(_encode_one, client, payload, prompt)
            else:
                fut = pool.submit(_decode_one, client, payload, prompt)
            futures[fut] = kind

        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            status = '✓' if result['ok'] else '✗'
            print(f'  [{i:>3}/{len(jobs)}] {status} {result["kind"]:6}  '
                  f'{result["elapsed"]:6.1f}s'
                  + (f'  ERR: {result["error"]}' if result.get('error') else ''))

    wall_elapsed = time.perf_counter() - wall_start

    _print_summary(results, wall_elapsed)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], wall_elapsed: float) -> None:
    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)

    for kind in ('encode', 'decode'):
        subset = [r for r in results if r['kind'] == kind]
        if not subset:
            continue
        ok = [r for r in subset if r['ok']]
        latencies = [r['elapsed'] for r in ok]
        print(f'\n{kind.upper()}  ({len(ok)}/{len(subset)} succeeded)')
        if latencies:
            print(f'  min   : {min(latencies):.1f}s')
            print(f'  median: {statistics.median(latencies):.1f}s')
            print(f'  p95   : {sorted(latencies)[int(len(latencies) * 0.95)]:.1f}s')
            print(f'  max   : {max(latencies):.1f}s')
        errors = [r for r in subset if not r['ok']]
        if errors:
            print(f'  errors ({len(errors)}):')
            for e in errors:
                print(f'    - {e.get("error")}')

    total_ok = sum(1 for r in results if r['ok'])
    print(f'\nTOTAL  {total_ok}/{len(results)} succeeded  '
          f'wall time: {wall_elapsed:.1f}s  '
          f'throughput: {len(results) / wall_elapsed:.2f} req/s')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stegosaurus load test')
    parser.add_argument(
        '--url', default='http://localhost:8080',
        help='Base URL of the running Stegosaurus app (default: http://localhost:8080)',
    )
    parser.add_argument(
        '--requests', type=int, default=10,
        help='Total number of requests to submit (default: 10)',
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of concurrent client threads (default: 4)',
    )
    parser.add_argument(
        '--prompt', default=_PROMPT,
        help='Prompt to use for all requests',
    )
    args = parser.parse_args()

    results = run(
        url=args.url,
        total_requests=args.requests,
        workers=args.workers,
        prompt=args.prompt,
    )

    failed = sum(1 for r in results if not r['ok'])
    sys.exit(1 if failed else 0)
