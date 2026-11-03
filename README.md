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

For live phone inference, open the dashboard over HTTPS, sign in, and choose **Start Live Tracking** in the mobile sensor tab. The browser requests motion permission when required, gathers overlapping 128 × 9 sensor windows at up to 50 Hz, and posts them to the authenticated `POST /predict` endpoint. That endpoint accepts `{"window": [[...9 channels...], ...128 timesteps...]}` and returns the CNN prediction, confidence, class probabilities, and inference latency. The existing channel-major `POST /predict/cnn` endpoint remains available.
