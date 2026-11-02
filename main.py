import os
import types

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
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)
