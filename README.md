<<<<<<< HEAD
# UCI HAR Dual-Pipeline AI API & Mobile Web Studio

**Team Members:** Tapash Pradhan (20240301067), Pushpa Ramtel, Shova Dhital  
**Department:** AI & Big Data Convergence, Seoul Christian University

---

## What's Included in This Package
1. **`app.py`**: Complete FastAPI backend serving:
   - `GET /` : Interactive Mobile & Desktop Web UI Dashboard
   - `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout` : Account registration and session management
   - `POST /predict/cnn` : 1D-CNN inference on raw `(9, 128)` sensor streams
   - `POST /predict/ann` : Regularized ANN inference on `561` engineered features
   - `POST /predict/compare` : Side-by-side comparison & 50/50 Softmax Ensemble
   - `GET /api/demo/{activity_name}` : Generate a preset and its comparison in one compact request
   - `GET /docs` : Interactive Swagger OpenAPI documentation
2. **`models.py`**: Exact PyTorch `HAR_1DCNN` and `RegularizedHAR_ANN` architectures.
3. **`static/index.html`**: Responsive dark-mode UI with Chart.js real-time waveforms, probability bar charts, preset simulators, and HTML5 Smartphone `DeviceMotionEvent` sensor streaming.
4. **`sample_generator.py`**: Built-in generator for realistic 6-class inertial waveforms so you can test immediately.

---

## Step-by-Step Setup Guide

### Step 1: Extract & Place Your Trained Weights
Unzip `HAR_API_Full_Package.zip` into a folder on your laptop, and place the two weight files you downloaded from Google Colab into the same root folder:
- `har_1dcnn_weights.pth`
- `har_ann_regularized.pth`
*(Note: The app also runs in a built-in kinematic demo mode even before you copy the `.pth` files!)*

### Step 2: Install Dependencies
Open your terminal (or VS Code terminal) in the project folder:
```bash
pip install -r requirements.txt
```

### Step 3: Run the API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Open on Your Laptop & Smartphone
- **On your Laptop:** Open `http://localhost:8000`
- **On your Smartphone (Same Wi-Fi):**
  1. Find your laptop's local IP address (`ipconfig` on Windows or `ifconfig` on Mac, e.g., `192.168.0.15`).
  2. Open `http://192.168.0.15:8000` on your phone browser.
  3. Tap **Preset Activity Simulator** to test all 6 activities, or switch to **Live Mobile Sensor Stream** to stream your phone's real accelerometer & gyroscope!

### Account Login
- On the dashboard, choose **Create an account** to register a username and a password of at least 12 characters. Registration signs you in automatically.
- User records and hashed passwords are stored in `har_auth.sqlite3` beside `app.py`. Set `HAR_AUTH_DB` to change the database path.
- Dashboard status, sample, and prediction endpoints require a valid session. Sessions expire after seven days; use **Sign out** to end a session.
- Preset simulator requests generate their sample and prediction together. Because generated presets do not contain real UCI 561-feature inputs, the dashboard skips ANN inference for them; use `/predict/compare` with real features for a trained ANN comparison.
- Preset samples are deterministic, making repeated activity selections consistent and avoiding changes to NumPy's shared random state.
- For deployment beyond a trusted local network, serve the app over HTTPS. The session cookie is HTTP-only and marked Secure for HTTPS requests.
- Same-origin use (including the phone dashboard) needs no CORS setup. For a separate browser client, set `HAR_ALLOWED_ORIGINS` to a comma-separated list of trusted origins, such as `https://dashboard.example.com`.
=======
---
title: Uci Har Studio
emoji: 🐠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: 'UCI Human Activity Recognition Using Smartphones '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 3339b9350d83a846be6df4fd1afbc7e0aa82b1cd
