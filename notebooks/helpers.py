'''Shared helpers for Stegosaurus scaling experiment notebooks.'''

import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from gradio_client import Client


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def save_data(filename: str, data) -> None:
    '''Save any JSON-serializable object to DATA_DIR/<filename>.'''
    path = DATA_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    n = len(data) if hasattr(data, '__len__') else ''
    print(f'Saved {n} to {path}')


def load_data(filename: str):
    '''Load a previously saved data file. Returns None if not found.'''
    path = DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

def make_client(app_url: str) -> Client:
    return Client(app_url, verbose=False)


def encode_one(app_url: str, prompt: str, message: str) -> float:
    '''Return wall-clock elapsed seconds for a single encode call.'''
    client = make_client(app_url)
    t0 = time.perf_counter()
    client.predict(prompt, message, api_name='/encode')
    return time.perf_counter() - t0


def decode_one(app_url: str, prompt: str, cover_text: str) -> float:
    '''Return wall-clock elapsed seconds for a single decode call.'''
    client = make_client(app_url)
    t0 = time.perf_counter()
    client.predict(prompt, cover_text, api_name='/decode')
    return time.perf_counter() - t0


def burst(app_url: str, prompt: str, message: str, n_requests: int, n_workers: int):
    '''Fire n_requests encode calls with n_workers threads in parallel.
    Returns (list_of_elapsed_times, total_wall_clock_seconds).'''

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(encode_one, app_url, prompt, message) for _ in range(n_requests)]
        times = [f.result() for f in as_completed(futures)]

    wall = time.perf_counter() - t_start
    return times, wall


def summarize(times: list, wall: float) -> dict:
    '''Return basic latency and throughput statistics for a burst run.'''
    sorted_t = sorted(times)
    n = len(sorted_t)

    return {
        'n': n,
        'min_s': round(sorted_t[0], 2),
        'median_s': round(statistics.median(sorted_t), 2),
        'p95_s': round(sorted_t[int(n * 0.95)], 2),
        'max_s': round(sorted_t[-1], 2),
        'throughput_rps': round(n / wall, 2),
    }


# ---------------------------------------------------------------------------
# Worker process counting (dynamic scaling trace)
# ---------------------------------------------------------------------------

def get_app_pid() -> str | None:
    '''Return the PID string of the running app.py process, or None.'''
    try:
        out = subprocess.check_output(['pgrep', '-f', r'python.*app\.py'], text=True)
        pids = out.strip().splitlines()
        return pids[0].strip() if pids else None
    except subprocess.CalledProcessError:
        return None


def count_workers() -> int:
    '''Count worker processes that are direct children of app.py.'''
    app_pid = get_app_pid()
    if not app_pid:
        return 0
    try:
        out = subprocess.check_output(
            ['ps', '--ppid', app_pid, '-o', 'cmd='], text=True
        )
        return sum(1 for line in out.splitlines() if 'spawn_main' in line)
    except subprocess.CalledProcessError:
        return 0


def sample_workers(stop_event, samples: list, sample_interval_s: float = 1.0):
    '''Background thread: append (timestamp, count) to samples until stop_event.'''
    t0 = time.perf_counter()
    while not stop_event.is_set():
        samples.append((time.perf_counter() - t0, count_workers()))
        time.sleep(sample_interval_s)


def run_sweep_step(
    data_key: str,
    run_flag: bool,
    app_url: str,
    prompt: str,
    message: str,
    n_requests: int,
    concurrency: int,
    extra_fields: dict | None = None,
    sort_by: str | None = None,
):
    '''Load or run one step of an incremental worker-count sweep.

    When run_flag is False: loads existing rows from scaling_data.json and
    returns a DataFrame (or None if no saved data is found).
    When run_flag is True: fires a burst, appends a new row (extra_fields +
    summarize stats), saves, and returns the updated DataFrame.
    '''
    _d = load_data('scaling_data.json') or {}

    if not run_flag:
        rows = _d.get(data_key)
        if rows:
            df = pd.DataFrame(rows)
            if sort_by:
                df = df.sort_values(sort_by).reset_index(drop=True)
            print(f'Loaded {len(df)} rows from scaling_data.json')
            return df
        print(f'run_flag=False and no saved data found for "{data_key}".')
        return None

    rows = _d.get(data_key) or []
    times, wall = burst(app_url, prompt, message, n_requests, concurrency)
    rows.append({**(extra_fields or {}), **summarize(times, wall)})
    _d[data_key] = rows
    save_data('scaling_data.json', _d)
    df = pd.DataFrame(rows)
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)
    return df


def run_scaling_trace(
    app_url: str,
    prompt: str,
    message: str,
    n_requests: int,
    concurrency: int,
    sample_interval_s: float = 1.0,
    quiesce_stable_s: float = 10.0,
    quiesce_timeout_s: float = 180.0,
):
    '''Fire a burst of requests while recording worker-process count over time.

    Starts a background sampler, waits 3 s to establish a baseline, fires
    the burst, then waits for the pool to quiesce back to baseline.

    Returns:
        samples   -- list of (elapsed_s, worker_count) tuples
        burst_wall -- wall-clock seconds the burst took
    '''
    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_workers,
        args=(stop_event, samples, sample_interval_s),
        daemon=True,
    )

    print('Sampling worker count before burst…')
    sampler.start()
    time.sleep(3)

    baseline_count = samples[-1][1] if samples else 1
    print(f'Baseline worker count: {baseline_count}')

    print(f'Firing burst: {n_requests} requests at concurrency {concurrency}')
    _, burst_wall = burst(app_url, prompt, message, n_requests, concurrency)
    print(f'Burst complete in {burst_wall:.1f}s. Waiting for pool to quiesce…')

    prev_count = None
    stable_since = time.perf_counter()
    quiesce_deadline = time.perf_counter() + quiesce_timeout_s

    while True:
        time.sleep(sample_interval_s)
        current = count_workers()

        if current != prev_count:
            stable_since = time.perf_counter()
            prev_count = current

        if current <= baseline_count and (time.perf_counter() - stable_since) >= quiesce_stable_s:
            break

        if time.perf_counter() >= quiesce_deadline:
            print(f'Warning: quiesce timeout reached; last count was {current} (baseline {baseline_count})')
            break

    stop_event.set()
    sampler.join()
    print(f'Done. Collected {len(samples)} samples.')
    return samples, burst_wall
