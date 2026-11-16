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

UCI HAR web app using the trained CNN-LSTM (`cnn_lstm_model.keras`) as the default live inference model, with the existing PyTorch 1D-CNN and ANN models available as alternatives. FastAPI is mounted via Gradio at `/gradio` with ZeroGPU support. The root route serves the static dashboard at `/static/index.html`.

For live phone inference, open the dashboard over HTTPS, sign in, and choose **Start Live Tracking** in the mobile sensor tab. The browser requests motion permission when required, gathers overlapping 128 × 9 sensor windows at up to 50 Hz, and posts them to the authenticated `POST /predict` endpoint. The default CNN-LSTM consumes those calibrated 128 × 9 windows and returns one of `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, or `LAYING`, along with confidence, class probabilities, and inference latency. The existing PyTorch CNN/ANN paths remain selectable, as does channel-major `POST /predict/cnn`. `loso_model.joblib` and `svm_model.joblib` are included model artifacts but are not loaded by the current inference endpoints.
