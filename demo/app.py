'''Gradio web interface for Stegosaurus encode/decode.'''

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from stegosaurus import encode, decode, PROMPT


def encode_message(prompt: str, message: str) -> str:

    if not message.strip():
        return 'Please enter a secret message.'

    return encode(message, prompt=prompt)


def decode_message(cover_text: str) -> str:

    if not cover_text.strip():
        return 'Please enter cover text to decode.'

    return decode(cover_text)


with gr.Blocks(title='Stegosaurus') as demo:

    gr.Markdown('# Stegosaurus\nHide secret messages inside AI-generated text.')

    with gr.Tabs():

        with gr.Tab('Encode'):

            prompt_input = gr.Textbox(
                label='Prompt',
                value=PROMPT,
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
            )

            encode_button.click(
                fn=encode_message,
                inputs=[prompt_input, message_input],
                outputs=cover_output,
            )

        with gr.Tab('Decode'):

            cover_input = gr.Textbox(
                label='Cover text',
                placeholder='Paste cover text to decode...',
                lines=8,
            )

            decode_button = gr.Button('Decode', variant='primary')

            message_output = gr.Textbox(
                label='Secret message',
                lines=2,
                interactive=False,
            )

            decode_button.click(
                fn=decode_message,
                inputs=cover_input,
                outputs=message_output,
            )


if __name__ == '__main__':
    demo.launch()
