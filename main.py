from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Import your existing FastAPI app from app.py
from app import app


@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
  return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=7860)