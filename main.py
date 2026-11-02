import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Import your existing FastAPI app from app.py
from app import app


# Root redirect to open your web dashboard
@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
  return RedirectResponse(url="/static/index.html")


# Create a simple Gradio dummy block so Hugging Face keeps the container alive
demo = gr.Interface(
    fn=lambda: "FastAPI Backend is Running!",
    inputs=None,
    outputs="text",
    title="UCI HAR Studio API",
    description=(
        "FastAPI backend and web dashboard are active. Open the full UI via"
        " API / static link."
    ),
)

# Mount Gradio on top of FastAPI if needed, or launch Uvicorn directly
if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=7860)