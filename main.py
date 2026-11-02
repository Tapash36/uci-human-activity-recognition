from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Import your existing FastAPI app from app.py
from app import app


# Automatically redirect the root URL to your static web dashboard
@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
  return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
  import uvicorn

  # Hugging Face Spaces strictly requires port 7860
  uvicorn.run(app, host="0.0.0.0", port=7860)