'''Gradio web interface for Stegosaurus encode/decode.'''

import atexit
import multiprocessing
import queue
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from stegosaurus import PROMPT
from job import Job
from manager import WorkerManager, parse_memory_limit

_DEFAULT_PROMPT = PROMPT
_JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 300))  # seconds

# These are set up in __main__ before the Gradio server starts
_job_queue = None
_ctx = None


def _submit(kind: str, args: dict) -> tuple[str, str]:
    '''Put a job on the queue and block until the worker returns a result.'''
    result_queue = _ctx.Queue()
    try:
        _job_queue.put_nowait(Job(kind=kind, args=args, result_queue=result_queue))
    except queue.Full:
        return '', 'Server is busy — please try again in a moment.'

    try:
        status, payload = result_queue.get(timeout=_JOB_TIMEOUT)
    except Exception:
        return '', 'Request timed out — the server may be overloaded.'

    if status == 'error':
        return '', f'Error: {payload}'

    return payload, ''


def encode_message(prompt: str, message: str) -> tuple[str, str]:

    if not message.strip():
        return '', 'Please enter a secret message.'

    return _submit('encode', {'message': message, 'prompt': prompt})


def decode_message(prompt: str, cover_text: str) -> tuple[str, str]:

    if not cover_text.strip():
        return '', 'Please enter cover text to decode.'

    return _submit('decode', {'cover_text': cover_text, 'prompt': prompt})


with gr.Blocks(title='Stegosaurus') as demo:

    gr.Markdown('# Stegosaurus\nHide secret messages inside AI-generated text. GitHub repository: [gperdrizet/stegosaurus](https://github.com/gperdrizet/stegosaurus)')

    with gr.Tabs():
        with gr.Tab('Encode'):

            prompt_input = gr.Textbox(
                label='Prompt',
                value=_DEFAULT_PROMPT,
                lines=2,
            )

            message_input = gr.Textbox(
                label='Secret message',
                placeholder='Enter the message to hide...',
                lines=2,
            )

            encode_button = gr.Button('Encode', variant='primary')
            encode_status = gr.Markdown('', visible=True)

            cover_output = gr.Textbox(
                label='Cover text',
                lines=10,
                interactive=False,
                elem_id='cover-output',
            )

            copy_button = gr.Button('Copy to clipboard', variant='secondary')

            encode_button.click(
                fn=lambda p, m: 'Encoding…',
                inputs=[prompt_input, message_input],
                outputs=encode_status,
                queue=False,
            ).then(
                fn=encode_message,
                inputs=[prompt_input, message_input],
                outputs=[cover_output, encode_status],
                show_progress='hidden',
            )

            copy_button.click(
                fn=None,
                inputs=cover_output,
                js='(text) => { navigator.clipboard.writeText(text); }',
            )

        with gr.Tab('Decode'):

            decode_prompt_input = gr.Textbox(
                label='Prompt',
                value=_DEFAULT_PROMPT,
                lines=2,
            )

            cover_input = gr.Textbox(
                label='Cover text',
                placeholder='Paste cover text to decode...',
                lines=3,
                max_lines=3,
            )

            decode_button = gr.Button('Decode', variant='primary')
            decode_status = gr.Markdown('', visible=True)

            message_output = gr.Textbox(
                label='Secret message',
                lines=2,
                interactive=False,
            )

            decode_button.click(
                fn=lambda p, c: 'Decoding…',
                inputs=[decode_prompt_input, cover_input],
                outputs=decode_status,
                queue=False,
            ).then(
                fn=decode_message,
                inputs=[decode_prompt_input, cover_input],
                outputs=[message_output, decode_status],
                show_progress='hidden',
            )


if __name__ == '__main__':

    # Must be set before any CUDA/torch code is imported in this process.
    # 'spawn' is required for CUDA; 'fork' causes deadlocks with GPU contexts.
    multiprocessing.set_start_method('spawn', force=True)
    _ctx = multiprocessing.get_context('spawn')

    _max_queue_size = int(os.environ.get('MAX_QUEUE_SIZE', 50))
    _job_queue = _ctx.Queue(maxsize=_max_queue_size)

    _manager = WorkerManager(
        ctx=_ctx,
        job_queue=_job_queue,
        max_memory_bytes=parse_memory_limit(os.environ.get('MAX_MEMORY', '0')),
        min_workers=int(os.environ.get('MIN_WORKERS', 1)),
        scale_interval=float(os.environ.get('SCALE_INTERVAL', 2.0)),
    )
    _manager.start()
    atexit.register(_manager.shutdown)

    print('Launching Stegosaurus demo...')

    demo.launch(
        server_name='0.0.0.0',
        server_port=int(os.environ.get('PORT', 8080)),
        root_path=os.environ.get('ROOT_PATH', ''),
    )

