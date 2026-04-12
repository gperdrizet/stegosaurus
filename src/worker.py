'''Worker process entry point for the Stegosaurus encode/decode pool.

Each worker is a spawned process that loads the model once, then loops
pulling jobs from job_queue and writing results back to each job's
per-job result_queue.  A None sentinel causes the worker to exit cleanly.
'''

import os
import sys
import logging

# Ensure src/ is importable when the module is run as a spawned process
sys.path.insert(0, os.path.dirname(__file__))


def _measure_memory_mb(device) -> float:
    '''Return current process memory footprint in MB.

    GPU: bytes allocated on the CUDA device.
    CPU: resident set size from /proc/self/status (Linux only; falls back to 0).
    '''
    if device.type == 'cuda':
        import torch
        return torch.cuda.memory_allocated(device) / (1024 ** 2)

    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # kB → MB
    except OSError:
        pass

    return 0.0


def run(job_queue, memory_report_queue, response_queue, threads_per_worker=None):
    '''Entry point executed in each worker process.

    Parameters
    ----------
    job_queue:
        Shared multiprocessing.Queue from which Job objects are consumed.
    memory_report_queue:
        Queue used to send the measured model footprint (float MB) back to
        the WorkerManager after the model has been loaded.
    response_queue:
        Shared multiprocessing.Queue to which the worker writes
        (correlation_id, status, payload) tuples.  The main process
        routes each tuple back to the correct waiting thread.
    threads_per_worker:
        Optional multiprocessing.Value("i") updated by WorkerManager after
        each scale event.  When set, torch.set_num_threads() is called before
        every job so each worker gets an equal share of CPU cores.
    '''

    # Each worker process has its own copy of the module; the module-level
    # model cache (_model, _tokenizer) is safe to use without locks.
    from stegosaurus import encode, decode, _load_model

    logger = logging.getLogger('stegosaurus.worker')

    logger.info('Worker pid=%d starting, loading model…', os.getpid())
    _, _, device = _load_model()
    footprint_mb = _measure_memory_mb(device)
    logger.info('Worker pid=%d model loaded (%.0f MB)', os.getpid(), footprint_mb)

    # Report actual footprint to WorkerManager so it can refine max_workers
    memory_report_queue.put(footprint_mb)

    while True:
        job = job_queue.get()

        # None sentinel signals graceful shutdown
        if job is None:
            logger.info('Worker pid=%d received shutdown sentinel', os.getpid())
            break

        if threads_per_worker is not None:
            try:
                import torch
                torch.set_num_threads(max(1, threads_per_worker.value))
            except Exception:
                pass

        try:
            if job.kind == 'encode':
                result = encode(**job.args)
            elif job.kind == 'decode':
                result = decode(**job.args)
            else:
                raise ValueError(f'Unknown job kind: {job.kind!r}')

            response_queue.put((job.correlation_id, 'ok', result))

        except Exception as exc:  # noqa: BLE001
            logger.exception('Worker pid=%d job failed', os.getpid())
            response_queue.put((job.correlation_id, 'error', str(exc)))

    logger.info('Worker pid=%d exiting', os.getpid())
