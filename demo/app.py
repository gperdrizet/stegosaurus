'''Gradio web interface for Stegosaurus encode/decode.'''

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from stegosaurus import encode, decode, PROMPT

_DEFAULT_PROMPT = PROMPT


def encode_message(prompt: str, message: str) -> tuple[str, str]:

    if not message.strip():
        return '', 'Please enter a secret message.'

    return encode(message, prompt=prompt), ''


def decode_message(prompt: str, cover_text: str) -> tuple[str, str]:

    if not cover_text.strip():
        return '', 'Please enter cover text to decode.'

    return decode(cover_text, prompt=prompt), ''


with gr.Blocks(title='Stegosaurus') as demo:

    gr.Markdown('# Stegosaurus\nHide secret messages inside AI-generated text.')

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

    print('Launching Stegosaurus demo...')

    demo.launch(
        server_name='0.0.0.0',
        server_port=int(os.environ.get('PORT', 8080))
    )
