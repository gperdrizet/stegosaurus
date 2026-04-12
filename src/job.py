'''Job dataclass shared between the Gradio app and worker processes.'''

from dataclasses import dataclass


@dataclass
class Job:
    kind: str            # 'encode' or 'decode'
    args: dict           # keyword arguments forwarded to encode() / decode()
    correlation_id: int  # unique ID; worker echoes it back via response_queue
