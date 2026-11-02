import gradio as gr
import spaces
import uvicorn

from app import app as fastapi_app


@spaces.GPU
def keep_alive():
    return "ZeroGPU backend active"


demo = gr.Interface(fn=keep_alive, inputs=None, outputs="text")

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)