'''Job dataclass shared between the Gradio app and worker processes.'''

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    kind: str        # 'encode' or 'decode'
    args: dict       # keyword arguments forwarded to encode() / decode()
    result_queue: Any = field(repr=False)  # Manager Queue proxy; worker writes (status, payload) here
