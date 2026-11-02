import os
import types

# Capture the target server port before overriding Gradio's internal env vars.
SERVER_PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))

# Disable Gradio SSR mode and move Gradio's internal ports away from 7860
# before importing gradio so mount_gradio_app never binds port 7860 during lifespan startup.
os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["GRADIO_SERVER_PORT"] = "7861"
os.environ["GRADIO_NODE_SERVER_PORT"] = "7862"

try:
    import spaces
except ImportError:
    spaces = types.SimpleNamespace(
        GPU=lambda fn=None, **_kwargs: (fn if fn is not None else lambda f: f)
    )

try:
    import gradio as gr
except ImportError:
    gr = None

import uvicorn
from app import app as fastapi_app


@spaces.GPU
def keep_alive():
    return "ZeroGPU backend active"


if gr is not None:
    with gr.Blocks(title="UCI HAR Studio") as demo:
        gr.Markdown("### UCI HAR Studio — ZeroGPU & FastAPI Health")
        status = gr.Textbox(label="Status", value="ZeroGPU backend active")
        gr.Button("Check ZeroGPU").click(
            fn=keep_alive,
            inputs=None,
            outputs=status,
        )

    demo.ssr_mode = False
    try:
        app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio", ssr_mode=False)
    except TypeError:
        app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
else:
    demo = None
    app = fastapi_app


def _notify_zerogpu_startup() -> None:
    try:
        from spaces import zero as spaces_zero

        if hasattr(spaces_zero, "startup"):
            spaces_zero.startup()
            print("[ZeroGPU] Startup report sent successfully.", flush=True)
    except Exception as exc:
        print(f"[ZeroGPU] Startup report skipped: {exc!r}", flush=True)


if __name__ == "__main__":
    _notify_zerogpu_startup()
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
