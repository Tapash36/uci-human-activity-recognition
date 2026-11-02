import gradio as gr
import spaces

from app import app as fastapi_app


@spaces.GPU
def keep_alive():
    return "ZeroGPU backend active"


with gr.Blocks() as demo:
    status = gr.Textbox(label="Status")
    gr.Button("Check ZeroGPU").click(
        fn=keep_alive,
        inputs=None,
        outputs=status,
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")