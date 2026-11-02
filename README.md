---
title: UCI HAR Studio
emoji: 🏃
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: main.py
pinned: false
license: mit
short_description: UCI Human Activity Recognition Using Smartphones
---
# uci-human-activity-recognition

UCI HAR web app running dual PyTorch models (1D-CNN & ANN) on FastAPI, mounted via Gradio at `/gradio` with ZeroGPU support. The root route serves the static dashboard at `/static/index.html`.
