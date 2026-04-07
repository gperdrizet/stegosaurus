'''Gradio web interface for Stegosaurus encode/decode.'''

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from stegosaurus import encode, decode, PROMPT

# BOS token prepended automatically — strip it from the user-facing default
_BOS = '<|endoftext|>'
_DEFAULT_PROMPT = PROMPT.removeprefix(_BOS)


def encode_message(prompt: str, message: str) -> str:

    if not message.strip():
        return 'Please enter a secret message.'

    # Always prepend the BOS token regardless of what the user typed
    full_prompt = _BOS + prompt
    return encode(message, prompt=full_prompt)


def decode_message(prompt: str, cover_text: str) -> str:

    if not cover_text.strip():
        return 'Please enter cover text to decode.'

    full_prompt = _BOS + prompt
    return decode(cover_text, prompt=full_prompt)


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
    
            cover_output = gr.Textbox(
                label='Cover text',
                lines=8,
                interactive=False,
                show_copy_button=True,
            )

            encode_button.click(
                fn=encode_message,
                inputs=[prompt_input, message_input],
                outputs=cover_output,
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

            message_output = gr.Textbox(
                label='Secret message',
                lines=2,
                interactive=False,
            )

            decode_button.click(
                fn=decode_message,
                inputs=[decode_prompt_input, cover_input],
                outputs=message_output,
            )


if __name__ == '__main__':
    demo.launch()
