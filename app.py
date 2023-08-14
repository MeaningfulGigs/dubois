import os
from typing import Optional, Tuple
from threading import Lock

import gradio as gr

from query_data import get_condense_prompt_qa_chain


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, inp: str, history: Optional[Tuple[str, str]], chain):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                chain = get_condense_prompt_qa_chain()
            # Set OpenAI key
            import openai

            openai.api_key = os.environ["OPENAI_API_KEY"]
            # Run chain and append input.
            output = chain({"question": inp})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()

block = gr.Blocks()

with block:
    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="User Input",
            placeholder="Ask anything...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    state = gr.State()
    agent_state = gr.State()

    submit.click(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state],
    )
    message.submit(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state],
    )

block.launch(debug=True)
