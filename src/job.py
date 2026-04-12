'''Job dataclass shared between the Gradio app and worker processes.'''

from dataclasses import dataclass, field
from multiprocessing.queues import Queue


@dataclass
class Job:
    kind: str           # 'encode' or 'decode'
    args: dict          # keyword arguments forwarded to encode() / decode()
    result_queue: Queue = field(repr=False)  # per-job queue; worker writes result here
