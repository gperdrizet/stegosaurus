'''WorkerManager: spawns and scales Stegosaurus encoder/decoder worker processes.

The manager runs as a background thread in the Gradio process.  It monitors
the shared job_queue depth and adjusts the worker pool size between
min_workers and max_workers.

max_workers is derived from MAX_MEMORY (env var) divided by the per-worker
model footprint.  The footprint is first estimated from model_config.json, then
refined once the first worker reports its actual measured footprint.
'''

import json
import logging
import multiprocessing
import os
import threading
import time

logger = logging.getLogger('stegosaurus.manager')


def parse_memory_limit(value: str) -> int:
    '''Parse a memory string to bytes.

    Accepts formats:
      "16GB"  "16 GB"  "16384MB"  "16384 MB"  "16384" (treated as MB)
    Returns 0 if value is empty, "0", or unparseable (meaning no limit).
    '''
    value = value.strip()
    if not value or value == '0':
        return 0

    upper = value.upper().replace(' ', '')

    if upper.endswith('GB'):
        return int(float(upper[:-2]) * 1024 ** 3)
    if upper.endswith('MB'):
        return int(float(upper[:-2]) * 1024 ** 2)

    # Bare number interpreted as MB
    try:
        return int(float(value)) * 1024 ** 2
    except ValueError:
        logger.warning('Could not parse MAX_MEMORY=%r, ignoring limit', value)
        return 0


class WorkerManager(threading.Thread):
    '''Background thread that manages a pool of worker processes.

    Parameters
    ----------
    ctx:
        multiprocessing context (must be 'spawn').
    job_queue:
        Shared queue from which worker processes consume Job objects.
    max_memory_bytes:
        Hard memory budget (bytes). 0 means use model_config estimate × 4.
    min_workers:
        Minimum number of always-running workers.
    scale_interval:
        Seconds between scaling checks.
    '''

    def __init__(
        self,
        ctx,
        job_queue,
        max_memory_bytes: int = 0,
        min_workers: int = 1,
        scale_interval: float = 2.0,
    ):
        super().__init__(name='WorkerManager', daemon=True)

        self._ctx = ctx
        self._job_queue = job_queue
        self._max_memory_bytes = max_memory_bytes
        self._min_workers = min_workers
        self._scale_interval = scale_interval

        self._workers: list[multiprocessing.Process] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Workers report their actual footprint here after loading
        self._memory_report_queue = ctx.Queue()

        # Initial max_workers estimate from model_config.json
        self._max_workers = self._estimate_max_workers()
        logger.info(
            'WorkerManager init: min=%d max=%d memory_budget=%s',
            self._min_workers,
            self._max_workers,
            f'{max_memory_bytes / 1024**2:.0f} MB' if max_memory_bytes else 'auto',
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shutdown(self, timeout: float = 60.0):
        '''Signal all workers to stop and wait for them to exit.'''
        self._stop_event.set()

        with self._lock:
            n = len(self._workers)

        logger.info('WorkerManager shutting down %d workers', n)
        for _ in range(n):
            self._job_queue.put(None)

        deadline = time.monotonic() + timeout
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            remaining = deadline - time.monotonic()
            w.join(timeout=max(0.0, remaining))
            if w.is_alive():
                logger.warning('Worker pid=%d did not exit cleanly, terminating', w.pid)
                w.terminate()

        logger.info('WorkerManager shutdown complete')

    # ------------------------------------------------------------------
    # Threading.Thread interface
    # ------------------------------------------------------------------

    def run(self):
        # Bring pool up to min_workers immediately
        for _ in range(self._min_workers):
            self._spawn_worker()

        while not self._stop_event.is_set():
            time.sleep(self._scale_interval)
            self._drain_memory_reports()
            self._scale()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_worker(self):
        p = self._ctx.Process(
            target=_worker_entry,
            args=(self._job_queue, self._memory_report_queue),
            daemon=True,
        )
        p.start()
        with self._lock:
            self._workers.append(p)
        logger.info('Spawned worker pid=%d (pool size now %d)', p.pid, len(self._workers))

    def _reap_dead_workers(self):
        with self._lock:
            alive = [w for w in self._workers if w.is_alive()]
            dead_count = len(self._workers) - len(alive)
            self._workers = alive

        if dead_count:
            logger.warning('%d worker(s) died unexpectedly', dead_count)

    def _scale(self):
        self._reap_dead_workers()

        with self._lock:
            alive = len(self._workers)

        try:
            qsize = self._job_queue.qsize()
        except NotImplementedError:
            # macOS doesn't support qsize(); treat as non-zero to be safe
            qsize = 1

        if qsize > alive and alive < self._max_workers:
            logger.debug('Scaling up: qsize=%d alive=%d max=%d', qsize, alive, self._max_workers)
            self._spawn_worker()

        elif qsize == 0 and alive > self._min_workers:
            logger.debug('Scaling down: qsize=%d alive=%d min=%d', qsize, alive, self._min_workers)
            self._job_queue.put(None)  # one idle worker will pick this up and exit

    def _drain_memory_reports(self):
        '''Consume any pending footprint reports and refine max_workers.'''
        while True:
            try:
                footprint_mb = self._memory_report_queue.get_nowait()
            except Exception:  # queue.Empty
                break

            footprint_bytes = footprint_mb * 1024 ** 2
            if self._max_memory_bytes and footprint_bytes > 0:
                new_max = max(1, int(self._max_memory_bytes / footprint_bytes))
                if new_max != self._max_workers:
                    logger.info(
                        'Refined max_workers from %d to %d '
                        '(footprint=%.0f MB, budget=%.0f MB)',
                        self._max_workers,
                        new_max,
                        footprint_mb,
                        self._max_memory_bytes / 1024 ** 2,
                    )
                    self._max_workers = new_max

    def _estimate_max_workers(self) -> int:
        '''Estimate max_workers from available memory and model_config.json.

        When MAX_MEMORY is set, use it directly.  When it is 0 (auto), query
        available memory from the hardware: VRAM via PyTorch, RAM via
        /proc/meminfo.  A 10% headroom is reserved in auto mode to avoid
        crowding out the OS and the Gradio process.  Falls back to 4 if
        neither the hardware query nor the config footprint can be read.
        '''
        _HEADROOM = 0.90  # use at most 90% of available memory in auto mode

        # --- determine memory budget ---
        if self._max_memory_bytes:
            budget_bytes = self._max_memory_bytes
        else:
            budget_bytes = self._available_memory_bytes()
            if budget_bytes:
                budget_bytes = int(budget_bytes * _HEADROOM)
                logger.info(
                    'Auto memory budget: %.0f MB (%.0f%% of available)',
                    budget_bytes / 1024 ** 2,
                    _HEADROOM * 100,
                )

        if not budget_bytes:
            return 4  # final fallback

        # --- determine per-worker footprint from config ---
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
            with open(config_path) as f:
                all_configs = json.load(f)
            model_name = os.environ.get('MODEL', 'Qwen/Qwen3-0.6B')
            memory_mb = all_configs.get(model_name, {}).get('memory_mb', 0)
            if memory_mb:
                return max(1, int(budget_bytes / (memory_mb * 1024 ** 2)))
        except Exception:
            pass

        return 4  # config unreadable fallback

    @staticmethod
    def _available_memory_bytes() -> int:
        '''Return available memory in bytes: VRAM if a CUDA device is present,
        otherwise free RAM from /proc/meminfo.  Returns 0 if undetectable.'''
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory
                reserved = torch.cuda.memory_reserved(device)
                return total - reserved
        except Exception:
            pass

        # CPU / RAM fallback
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024  # kB → bytes
        except OSError:
            pass

        return 0


def _worker_entry(job_queue, memory_report_queue):
    '''Module-level function so it can be pickled by the spawn start method.'''
    from worker import run
    run(job_queue, memory_report_queue)
