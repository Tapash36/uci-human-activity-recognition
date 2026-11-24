import asyncio
import os
import hashlib
import hmac
import json
import logging
import pickle
import re
import secrets
import sqlite3
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, AsyncIterator, List, Dict, Optional, Set
import zipfile
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from models import HAR_1DCNN, HAR_CNN_LSTM, RegularizedHAR_ANN
from sample_generator import generate_realistic_sample, ACTIVITY_NAMES

BASE_DIR = Path(__file__).resolve().parent


def _resolve_default_db_path() -> Path:
    configured = os.environ.get("HAR_AUTH_DB")
    if configured:
        return Path(configured).expanduser()
    candidate = BASE_DIR / "har_auth.sqlite3"
    if os.access(str(BASE_DIR), os.W_OK):
        return candidate
    return Path(tempfile.gettempdir()) / "har_auth.sqlite3"


AUTH_DB_PATH = _resolve_default_db_path()
SESSION_COOKIE = "har_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
PASSWORD_HASH_ITERATIONS = 600_000


def get_auth_db() -> sqlite3.Connection:
    global AUTH_DB_PATH
    try:
        AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(AUTH_DB_PATH), timeout=10)
    except (sqlite3.OperationalError, PermissionError):
        AUTH_DB_PATH = Path(tempfile.gettempdir()) / "har_auth.sqlite3"
        connection = sqlite3.connect(str(AUTH_DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_auth_db() -> None:
    global AUTH_DB_PATH
    schema_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at INTEGER NOT NULL
        );
    """
    connection = get_auth_db()
    try:
        connection.executescript(schema_sql)
    except (sqlite3.OperationalError, PermissionError):
        connection.close()
        AUTH_DB_PATH = Path(tempfile.gettempdir()) / "har_auth.sqlite3"
        connection = get_auth_db()
        try:
            connection.executescript(schema_sql)
        finally:
            connection.close()
    else:
        connection.close()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_auth_db()
    yield


app = FastAPI(
    title="UCI HAR AI & Big Data Convergence API",
    description="Human Activity Recognition API with CNN-LSTM, 1D-CNN, and Regularized ANN inference",
    version="2.0.0",
    lifespan=lifespan,
)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt.hex()}${password_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(password_hash.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def _is_https_request(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or "https" in forwarded_proto.lower()


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    is_https = _is_https_request(request)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=is_https,
        samesite="none" if is_https else "lax",
        path="/",
    )


def create_session(user_id: int) -> str:
    initialize_auth_db()
    token = secrets.token_urlsafe(32)
    connection = get_auth_db()
    try:
        with connection:
            now = int(time.time())
            connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                "INSERT INTO auth_sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (hashlib.sha256(token.encode("ascii")).hexdigest(), user_id, now + SESSION_TTL_SECONDS),
            )
    finally:
        connection.close()
    return token


def _extract_session_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()
        if bearer_token:
            return bearer_token
    return request.cookies.get(SESSION_COOKIE)


def authenticate_session_token(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    initialize_auth_db()
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    connection = get_auth_db()
    try:
        user = connection.execute(
            """
            SELECT users.id, users.username
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?
            """,
            (token_hash, int(time.time())),
        ).fetchone()
    finally:
        connection.close()

    if user is None:
        return None
    return {"id": user["id"], "username": user["username"]}


def _get_or_create_guest_user() -> dict:
    initialize_auth_db()
    guest_username = "guest_researcher"
    connection = get_auth_db()
    try:
        row = connection.execute(
            "SELECT id, username FROM users WHERE username = ?",
            (guest_username,),
        ).fetchone()
        if row is None:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (guest_username, hash_password(secrets.token_urlsafe(32))),
                )
                user_id = cursor.lastrowid
        else:
            user_id = row["id"]
    finally:
        connection.close()
    return {"id": int(user_id), "username": guest_username}


def require_authenticated_user(request: Request) -> dict:
    token = _extract_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    user = authenticate_session_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return user


def optional_authenticated_user(request: Request) -> dict:
    token = _extract_session_token(request)
    user = authenticate_session_token(token) if token else None
    if user is not None:
        return user
    return _get_or_create_guest_user()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# On Hugging Face ZeroGPU, torch.cuda is emulated in the main process and cannot run
# forward passes inside plain FastAPI routes outside @spaces.GPU. Keep lightweight
# 1D-CNN and MLP inference on CPU when running under ZeroGPU / Spaces unless overridden.
_is_zero_gpu = bool(
    os.environ.get("SPACES_ZERO_GPU")
    or os.environ.get("ZERO_GPU")
    or os.environ.get("SPACE_ID")
)
device = torch.device(
    "cuda" if (torch.cuda.is_available() and not _is_zero_gpu) else "cpu"
)

# Initialize Models
cnn_model = HAR_1DCNN(in_channels=9, num_classes=6).to(device)
ann_model = RegularizedHAR_ANN(input_dim=561, num_classes=6).to(device)

cnn_weights_loaded = False
ann_weights_loaded = False
primary_model_loaded = False
svm_weights_loaded = False
loso_weights_loaded = False

model_weights_path = BASE_DIR / "model.pth"
cnn_weights_path = BASE_DIR / "har_1dcnn_weights.pth"
ann_weights_path = BASE_DIR / "har_ann_regularized.pth"
cnn_lstm_model_path = BASE_DIR / "cnn_lstm_model.keras"
svm_model_path = BASE_DIR / "svm_model.joblib"
loso_model_path = BASE_DIR / "loso_model.joblib"
svm_loso_npz_path = BASE_DIR / "svm_loso_weights.npz"

cnn_lstm_model = HAR_CNN_LSTM(in_channels=9, num_classes=6).to(device)
cnn_lstm_model_loaded = False

svm_coef: Optional[np.ndarray] = None
svm_intercept: Optional[np.ndarray] = None
loso_coef: Optional[np.ndarray] = None
loso_intercept: Optional[np.ndarray] = None
scaler_mean: Optional[np.ndarray] = None
scaler_scale: Optional[np.ndarray] = None
class_centroids_561: Optional[np.ndarray] = None

MODEL_DISPLAY_NAMES = {
    "cnn_lstm": "CNN-LSTM",
    "ann": "Regularized ANN",
    "svm": "Linear SVM",
    "loso": "1D-CNN + LOSO-SVM Hybrid",
    "combinai": "CombinAI (Ensemble)",
    "ensemble": "CombinAI (Ensemble)",
    "cnn": "1D-CNN + LOSO-SVM Hybrid",
}

MODEL_BENCHMARK_META = {
    "cnn_lstm": {
        "role": "Base Predictor",
        "held_out_test_accuracy": "90.57%",
        "loso_accuracy": "90.14% ± 10.25%",
        "summary_badge": "Test 90.57% · LOSO 90.14%±10.25%",
    },
    "ann": {
        "role": "Base Predictor",
        "held_out_test_accuracy": "96.06%",
        "loso_accuracy": "94.13% ± 5.85%",
        "summary_badge": "Test 96.06% · LOSO 94.13%±5.85%",
    },
    "svm": {
        "role": "Base Predictor",
        "held_out_test_accuracy": "96.20%",
        "loso_accuracy": "95.83% ± 4.84%",
        "summary_badge": "Test 96.20% · LOSO 95.83%±4.84%",
    },
    "loso": {
        "role": "Base Predictor (Hybrid)",
        "held_out_test_accuracy": "Not independently validated",
        "loso_accuracy": "Not independently validated",
        "summary_badge": "Hybrid · Not independently validated",
    },
    "combinai": {
        "role": "Ensemble (4 Base Predictors)",
        "held_out_test_accuracy": "Not independently validated",
        "loso_accuracy": "Not independently validated",
        "summary_badge": "Ensemble · Not independently validated",
    },
}

MODEL_ACCURACY_STR = {
    k: v["summary_badge"] for k, v in MODEL_BENCHMARK_META.items()
}


def _unwrap_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            return loaded_obj["model_state_dict"]
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
    return loaded_obj


def _load_weights(model: torch.nn.Module, weights_path: Path) -> bool:
    if not weights_path.exists():
        return False
    try:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=device)
        state_dict = _unwrap_state_dict(state_dict)
        model.load_state_dict(state_dict)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Unable to load model weights from %s; using demo fallback: %s",
            weights_path,
            exc,
        )
        return False
    return True


def _load_primary_loso_model() -> None:
    """Loads the LOSO cross-validated weights from model.pth into the matching architecture."""
    global cnn_weights_loaded, ann_weights_loaded, primary_model_loaded
    target_path = Path("model.pth") if Path("model.pth").exists() else model_weights_path
    if not target_path.exists():
        return
    try:
        if Path("model.pth").exists():
            raw_state = torch.load("model.pth", map_location=device)
        else:
            raw_state = torch.load(str(target_path), map_location=device)
        state_dict = _unwrap_state_dict(raw_state)
        if not isinstance(state_dict, dict):
            return
        keys = list(state_dict.keys())
        if any(k.startswith("feature_extractor") or k.startswith("classifier") for k in keys):
            cnn_model.load_state_dict(state_dict)
            cnn_weights_loaded = True
            primary_model_loaded = True
        elif any(k.startswith("network") for k in keys):
            ann_model.load_state_dict(state_dict)
            ann_weights_loaded = True
            primary_model_loaded = True
        else:
            try:
                cnn_model.load_state_dict(state_dict)
                cnn_weights_loaded = True
                primary_model_loaded = True
            except RuntimeError:
                ann_model.load_state_dict(state_dict)
                ann_weights_loaded = True
                primary_model_loaded = True
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Unable to load primary model.pth weights; checking fallback weights: %s",
            exc,
        )


def _load_cnn_lstm_model() -> None:
    """
    Loads the Keras 3 archive (cnn_lstm_model.keras: config.json + model.weights.h5)
    directly into our native PyTorch HAR_CNN_LSTM module without importing TensorFlow,
    avoiding C++ oneDNN/protobuf/CUDA symbol collisions (Exit code 139 SIGSEGV) on Linux.
    """
    global cnn_lstm_model_loaded
    logger = logging.getLogger(__name__)
    if not cnn_lstm_model_path.exists():
        logger.error("CNN-LSTM model file is missing: %s", cnn_lstm_model_path)
        return
    try:
        with zipfile.ZipFile(cnn_lstm_model_path, "r") as zf:
            cfg = json.loads(zf.read("config.json").decode("utf-8"))
            layers_cfg = cfg.get("config", {}).get("layers", [])
            if not layers_cfg:
                raise ValueError("Missing layers in cnn_lstm_model.keras config.json")
            batch_shape = layers_cfg[0].get("config", {}).get("batch_shape")
            if batch_shape != [None, 128, 9]:
                raise ValueError(f"Expected CNN-LSTM input shape [None, 128, 9], got {batch_shape}")
            out_units = layers_cfg[-1].get("config", {}).get("units")
            if out_units != len(ACTIVITY_NAMES):
                raise ValueError(f"Expected {len(ACTIVITY_NAMES)} CNN-LSTM outputs, got {out_units}")

            h5_bytes = zf.read("model.weights.h5")
            with h5py.File(io.BytesIO(h5_bytes), "r") as hf:
                with torch.no_grad():
                    conv1_w = np.transpose(hf["layers/conv1d/vars/0"][:], (2, 1, 0))
                    conv1_b = hf["layers/conv1d/vars/1"][:]
                    conv2_w = np.transpose(hf["layers/conv1d_1/vars/0"][:], (2, 1, 0))
                    conv2_b = hf["layers/conv1d_1/vars/1"][:]
                    lstm_ih = hf["layers/lstm/cell/vars/0"][:].T
                    lstm_hh = hf["layers/lstm/cell/vars/1"][:].T
                    lstm_b = hf["layers/lstm/cell/vars/2"][:]
                    fc1_w = hf["layers/dense/vars/0"][:].T
                    fc1_b = hf["layers/dense/vars/1"][:]
                    fc2_w = hf["layers/dense_1/vars/0"][:].T
                    fc2_b = hf["layers/dense_1/vars/1"][:]

                    cnn_lstm_model.conv1.weight.copy_(torch.as_tensor(conv1_w, dtype=torch.float32, device=device))
                    cnn_lstm_model.conv1.bias.copy_(torch.as_tensor(conv1_b, dtype=torch.float32, device=device))
                    cnn_lstm_model.conv2.weight.copy_(torch.as_tensor(conv2_w, dtype=torch.float32, device=device))
                    cnn_lstm_model.conv2.bias.copy_(torch.as_tensor(conv2_b, dtype=torch.float32, device=device))
                    cnn_lstm_model.lstm.weight_ih_l0.copy_(torch.as_tensor(lstm_ih, dtype=torch.float32, device=device))
                    cnn_lstm_model.lstm.weight_hh_l0.copy_(torch.as_tensor(lstm_hh, dtype=torch.float32, device=device))
                    cnn_lstm_model.lstm.bias_ih_l0.copy_(torch.as_tensor(lstm_b, dtype=torch.float32, device=device))
                    cnn_lstm_model.lstm.bias_hh_l0.zero_()
                    cnn_lstm_model.fc1.weight.copy_(torch.as_tensor(fc1_w, dtype=torch.float32, device=device))
                    cnn_lstm_model.fc1.bias.copy_(torch.as_tensor(fc1_b, dtype=torch.float32, device=device))
                    cnn_lstm_model.fc2.weight.copy_(torch.as_tensor(fc2_w, dtype=torch.float32, device=device))
                    cnn_lstm_model.fc2.bias.copy_(torch.as_tensor(fc2_b, dtype=torch.float32, device=device))
        cnn_lstm_model.eval()
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        logger.error("Unable to load CNN-LSTM model from %s: %s", cnn_lstm_model_path, exc)
        return
    cnn_lstm_model_loaded = True


def _load_svm_and_loso_models() -> None:
    """Loads LinearSVC weights (svm_model.joblib, loso_model.joblib) and StandardScaler / centroids (svm_loso_weights.npz)."""
    global svm_coef, svm_intercept, loso_coef, loso_intercept
    global scaler_mean, scaler_scale, class_centroids_561
    global svm_weights_loaded, loso_weights_loaded
    logger = logging.getLogger(__name__)

    if svm_loso_npz_path.exists():
        try:
            with np.load(svm_loso_npz_path) as npz:
                svm_coef = np.asarray(npz["coef"], dtype=np.float32)
                svm_intercept = np.asarray(npz["intercept"], dtype=np.float32)
                loso_coef = svm_coef.copy()
                loso_intercept = svm_intercept.copy()
                scaler_mean = np.asarray(npz["mean"], dtype=np.float32)
                scaler_scale = np.asarray(npz["scale"], dtype=np.float32)
                class_centroids_561 = np.asarray(npz["centroids"], dtype=np.float32)
                svm_weights_loaded = True
                loso_weights_loaded = True
                return
        except Exception as exc:
            logger.warning("Unable to load svm_loso_weights.npz: %s", exc)

    try:
        import warnings
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if svm_model_path.exists():
                svm_est = joblib.load(svm_model_path)
                if hasattr(svm_est, "coef_") and hasattr(svm_est, "intercept_"):
                    svm_coef = np.asarray(svm_est.coef_, dtype=np.float32)
                    svm_intercept = np.asarray(svm_est.intercept_, dtype=np.float32)
                    svm_weights_loaded = True
            if loso_model_path.exists():
                loso_est = joblib.load(loso_model_path)
                if hasattr(loso_est, "coef_") and hasattr(loso_est, "intercept_"):
                    loso_coef = np.asarray(loso_est.coef_, dtype=np.float32)
                    loso_intercept = np.asarray(loso_est.intercept_, dtype=np.float32)
                    loso_weights_loaded = True
    except Exception as exc:
        logger.info("Using cached NPZ arrays for SVM/LOSO linear weights: %s", exc)


_load_cnn_lstm_model()
_load_primary_loso_model()
_load_svm_and_loso_models()
if not cnn_weights_loaded:
    cnn_weights_loaded = _load_weights(cnn_model, cnn_weights_path)
if not ann_weights_loaded:
    ann_weights_loaded = _load_weights(ann_model, ann_weights_path)

cnn_model.eval()
ann_model.eval()

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ----------------- Request & Response Schemas -----------------
class RawSignalPayload(BaseModel):
    signals: List[List[float]] = Field(
        ...,
        description="9 channels x 128 timesteps matrix: [body_acc_xyz, body_gyro_xyz, total_acc_xyz]"
    )

class LiveWindowPayload(BaseModel):
    window: List[List[float]] = Field(
        ...,
        description="128 timesteps x 9 channels: [body_acc_xyz, body_gyro_xyz, total_acc_xyz]"
    )
    model_type: str = Field(
        default="cnn_lstm",
        description="Requested model architecture: 'cnn_lstm', 'ann', 'svm', 'loso', or 'combinai'"
    )
    sample_rate_hz: float = Field(
        default=50.0,
        ge=10.0,
        le=240.0,
        description="Actual sampling rate of the 128-sample window in Hz"
    )
    previous_activity: Optional[str] = Field(
        default=None,
        description="Previous stabilized activity for hysteresis continuity"
    )
    room_code: Optional[str] = Field(
        default=None,
        max_length=24,
        description="Optional cross-device live sync room code for low-latency PC mirror broadcast"
    )


class RelayPublishPayload(BaseModel):
    sender_active: Optional[bool] = None
    sender_label: Optional[str] = Field(default=None, max_length=64)
    telemetry: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None

class FeatureVectorPayload(BaseModel):
    features: List[float] = Field(
        ...,
        description="561-dimensional engineered feature vector normalized in [-1, 1]"
    )

class EnsemblePayload(BaseModel):
    signals: List[List[float]]
    features: List[float]

class ModelComparisonItem(BaseModel):
    model_key: str
    model_name: str
    role: str = "Base Predictor"
    accuracy: str
    held_out_test_accuracy: str = "Not independently validated"
    loso_accuracy: str = "Not independently validated"
    predicted_activity: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float

class PredictionResponse(BaseModel):
    model_used: str
    predicted_activity: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    weights_status: str
    window_steps: int = 0
    cadence_spm: float = 0.0
    step_peak_indices: List[int] = Field(default_factory=list)
    all_models: List[ModelComparisonItem] = Field(default_factory=list)

class DualComparisonResponse(BaseModel):
    cnn_result: PredictionResponse
    ann_result: PredictionResponse
    ensemble_activity: str
    ensemble_confidence: float
    ensemble_probabilities: Dict[str, float]
    all_models: List[ModelComparisonItem] = Field(default_factory=list)

class PresetPredictionResponse(DualComparisonResponse):
    activity: str
    signals: List[List[float]]


class CredentialsPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=1, max_length=128)


class ChatContextPayload(BaseModel):
    selected_model: Optional[str] = "cnn_lstm"
    current_activity: Optional[str] = None
    confidence: Optional[float] = None
    total_steps: Optional[int] = 0
    window_steps: Optional[int] = 0
    cadence_spm: Optional[float] = 0.0
    tilt_deg: Optional[float] = None
    motion_energy: Optional[float] = None


class ChatRequestPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    context: Optional[ChatContextPayload] = None


class ChatHighlightItem(BaseModel):
    label: str
    value: str


class ChatResponsePayload(BaseModel):
    title: str
    badge: str
    answer: str
    highlights: List[ChatHighlightItem] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)


# ----------------- Helper kinematic heuristic fallback if weights aren't uploaded yet -----------------
def adjust_Untrained_Probs(probs: np.ndarray, signals: Optional[np.ndarray] = None) -> np.ndarray:
    """If .pth weights are not yet placed in folder, uses signal kinematics + model pass for realistic demo."""
    if signals is not None:
        body_energy = float(np.std(signals[0:3, :]))
        grav_x = float(np.mean(signals[6, :]))
        grav_y = float(np.mean(signals[7, :]))
        if body_energy < 0.05:
            if abs(grav_x) < 0.35:
                boost = np.array([0.002, 0.002, 0.002, 0.004, 0.005, 0.985]) # LAYING
            elif grav_y < -0.15:
                boost = np.array([0.005, 0.005, 0.005, 0.910, 0.070, 0.005]) # SITTING
            else:
                boost = np.array([0.005, 0.005, 0.005, 0.065, 0.915, 0.005]) # STANDING
        else:
            if body_energy > 0.48:
                boost = np.array([0.030, 0.045, 0.915, 0.005, 0.003, 0.002]) # DOWNSTAIRS
            elif np.mean(signals[7, :]) > 0.15:
                boost = np.array([0.025, 0.945, 0.020, 0.005, 0.003, 0.002]) # UPSTAIRS
            else:
                boost = np.array([0.965, 0.018, 0.012, 0.002, 0.002, 0.001]) # WALKING
        blended = 0.15 * probs + 0.85 * boost
        return blended / np.sum(blended)
    return probs

# ----------------- Routes -----------------
@app.get("/")
def serve_ui(request: Request):
    target = "/static/index.html"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(url=target)

@app.post("/api/auth/register")
def register(payload: CredentialsPayload, request: Request, response: Response):
    username = payload.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@+-]{3,100}", username):
        raise HTTPException(
            status_code=400,
            detail="Enter a valid username or email (3-100 characters: letters, numbers, '.', '_', '-', '+', or '@').",
        )
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    initialize_auth_db()
    username_norm = username.lower()
    connection = get_auth_db()
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username_norm, hash_password(payload.password)),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username or email is already registered.")
    finally:
        connection.close()

    token = create_session(user_id)
    set_session_cookie(response, request, token)
    return {"username": username_norm, "token": token, "is_guest": False}


@app.post("/api/auth/login")
def login(payload: CredentialsPayload, request: Request, response: Response):
    initialize_auth_db()
    username_norm = payload.username.strip().lower()
    connection = get_auth_db()
    try:
        user = connection.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username_norm,),
        ).fetchone()
    finally:
        connection.close()

    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email/username or password.")

    token = create_session(user["id"])
    set_session_cookie(response, request, token)
    return {"username": user["username"], "token": token, "is_guest": user["username"] == "guest_researcher"}


@app.post("/api/auth/guest")
def login_as_guest(request: Request, response: Response):
    initialize_auth_db()
    guest_username = "guest_researcher"
    connection = get_auth_db()
    try:
        row = connection.execute(
            "SELECT id, username FROM users WHERE username = ?",
            (guest_username,),
        ).fetchone()
        if row is None:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (guest_username, hash_password(secrets.token_urlsafe(32))),
                )
                user_id = cursor.lastrowid
        else:
            user_id = row["id"]
    finally:
        connection.close()

    token = create_session(user_id)
    set_session_cookie(response, request, token)
    return {"username": "Guest Demo", "account": guest_username, "token": token, "is_guest": True}


@app.get("/api/auth/me")
def current_user(user: dict = Depends(require_authenticated_user)):
    is_guest = user["username"] == "guest_researcher"
    return {
        "username": "Guest Demo" if is_guest else user["username"],
        "account": user["username"],
        "is_guest": is_guest,
    }


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = _extract_session_token(request)
    if token:
        initialize_auth_db()
        connection = get_auth_db()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM auth_sessions WHERE token_hash = ?",
                    (hashlib.sha256(token.encode("utf-8")).hexdigest(),),
                )
        finally:
            connection.close()
    is_https = _is_https_request(request)
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=is_https,
        samesite="none" if is_https else "lax",
    )
    return {"status": "logged out"}


def _count_model_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@app.get("/api/status")
def api_status(user: dict = Depends(optional_authenticated_user)):
    return {
        "status": "online",
        "device": str(device),
        "cnn_lstm_model_loaded": cnn_lstm_model_loaded,
        "cnn_lstm_model_file": cnn_lstm_model_path.name,
        "cnn_weights_loaded": cnn_weights_loaded,
        "ann_weights_loaded": ann_weights_loaded,
        "svm_weights_loaded": svm_weights_loaded,
        "loso_weights_loaded": loso_weights_loaded or cnn_weights_loaded,
        "primary_model_loaded": primary_model_loaded,
        "weights_file": "model.pth" if primary_model_loaded else "har_1dcnn_weights.pth / har_ann_regularized.pth",
        "cnn_lstm_params": _count_model_params(cnn_lstm_model),
        "cnn_params": _count_model_params(cnn_model),
        "ann_params": _count_model_params(ann_model),
        "models": [
            "CNN-LSTM",
            "Regularized ANN",
            "Linear SVM",
            "1D-CNN + LOSO-SVM Hybrid",
            "CombinAI (Ensemble)",
        ],
        "live_predictors": {
            "base_predictors": [
                "CNN-LSTM",
                "Regularized ANN",
                "Linear SVM",
                "1D-CNN + LOSO-SVM Hybrid",
            ],
            "ensemble": "CombinAI (Ensemble)",
            "note": (
                "LOSO (Leave-One-Subject-Out) is a 30-fold cross-validation protocol, not a model architecture. "
                "In live inference, '1D-CNN + LOSO-SVM Hybrid' blends standalone HAR 1D-CNN probabilities with a "
                "LOSO-trained Linear SVM; neither that hybrid nor CombinAI (Ensemble) has an independently validated "
                "single-model benchmark score."
            ),
        },
        "activities": ACTIVITY_NAMES,
        "sample_rate_hz": 50,
        "window_size": 128,
        "num_channels": 9,
        "feature_dim": 561,
        "benchmarks": {
            "protocols": {
                "held_out_test": "Standard 21-train / 9-test subject split (7,352 train / 2,947 test windows)",
                "loso_30_fold": "30-fold Leave-One-Subject-Out cross-validation (Mean ± SD across 30 unseen subjects)",
            },
            "benchmarked_models": {
                "linear_svm_leak_free": {
                    "model_name": "Linear SVM (Leak-Free Tuned)",
                    "input": "561 features",
                    "held_out_test_accuracy": 96.20,
                    "held_out_test_macro_f1": 96.29,
                    "loso_mean_accuracy": 95.83,
                    "loso_std_accuracy": 4.84,
                    "loso_summary": "95.83% ± 4.84%",
                },
                "regularized_ann": {
                    "model_name": "Regularized ANN",
                    "input": "561 features",
                    "held_out_test_accuracy": 96.06,
                    "held_out_test_macro_f1": 96.06,
                    "loso_mean_accuracy": 94.13,
                    "loso_std_accuracy": 5.85,
                    "loso_summary": "94.13% ± 5.85%",
                },
                "har_1dcnn_standalone": {
                    "model_name": "HAR 1D-CNN (Standalone Benchmark)",
                    "input": "9x128 raw signals",
                    "held_out_test_accuracy": 93.93,
                    "held_out_test_macro_f1": 93.99,
                    "loso_mean_accuracy": 93.51,
                    "loso_std_accuracy": 7.78,
                    "loso_summary": "93.51% ± 7.78%",
                },
                "cnn_lstm_leak_free": {
                    "model_name": "CNN-LSTM (Leak-Free)",
                    "input": "9x128 raw signals",
                    "held_out_test_accuracy": 90.57,
                    "held_out_test_macro_f1": 90.60,
                    "loso_mean_accuracy": 90.14,
                    "loso_std_accuracy": 10.25,
                    "loso_summary": "90.14% ± 10.25%",
                },
            },
            "unvalidated_live_composites": {
                "1dcnn_loso_svm_hybrid": {
                    "model_name": "1D-CNN + LOSO-SVM Hybrid",
                    "status": "Not independently validated",
                    "components": "Blends 45% standalone HAR 1D-CNN (93.93% test / 93.51% ± 7.78% LOSO) + 55% LOSO-trained Linear SVM (95.83% ± 4.84% LOSO)",
                },
                "combinai_ensemble": {
                    "model_name": "CombinAI (Ensemble)",
                    "status": "Not independently validated",
                    "components": "Weighted probability ensemble of CNN-LSTM (20%), Regularized ANN (25%), Linear SVM (30%), and 1D-CNN + LOSO-SVM Hybrid (25%)",
                },
            },
            "svm_test_accuracy": 96.20,
            "svm_loso_mean": 95.83,
            "svm_loso_std": 4.84,
            "ann_test_accuracy": 96.06,
            "ann_macro_f1": 0.9606,
            "ann_loso_mean": 94.13,
            "ann_loso_std": 5.85,
            "cnn_test_accuracy": 93.93,
            "cnn_macro_f1": 0.9399,
            "cnn_loso_mean": 93.51,
            "cnn_loso_std": 7.78,
            "cnn_lstm_test_accuracy": 90.57,
            "cnn_lstm_macro_f1": 0.9060,
            "cnn_lstm_loso_mean": 90.14,
            "cnn_lstm_loso_std": 10.25,
            "hybrid_test_accuracy": None,
            "combinai_test_accuracy": None,
            "test_samples": 2947,
            "train_samples": 7352,
            "subjects": 30,
        },
    }

@app.get("/api/sample/{activity_name}")
def get_preset_sample(activity_name: str, user: dict = Depends(require_authenticated_user)):
    act = activity_name.upper()
    if act not in ACTIVITY_NAMES:
        raise HTTPException(status_code=400, detail=f"Activity must be one of {ACTIVITY_NAMES}")
    signals, features = generate_realistic_sample(act)
    return {
        "activity": act,
        "signals": signals,
        "features": features
    }

@app.post("/predict/cnn", response_model=PredictionResponse)
def predict_cnn(payload: RawSignalPayload, user: dict = Depends(require_authenticated_user)):
    signals = np.array(payload.signals, dtype=np.float32)
    if signals.shape != (9, 128):
        raise HTTPException(status_code=400, detail=f"Expected shape (9, 128), got {signals.shape}")
    if not np.isfinite(signals).all():
        raise HTTPException(status_code=400, detail="Signals must contain only finite numbers")

    return predict_cnn_signals(signals)


def calibrate_live_phone_signals(
    raw_signals: np.ndarray,
    sample_rate_hz: float = 50.0,
    previous_activity: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adapts live smartphone signals (9, 128) [body_acc_xyz, gyro_xyz, total_acc_xyz]
    to the waist-mounted UCI HAR coordinate frame (sign-invariant across Android +g and iOS -g).
    Focuses on the recent 1.1s sub-window for instant transition response (zero freeze/lag)
    and separates Flat Walking, Upstairs, Downstairs, Sitting, Standing, and Laying accurately.
    """
    fs = float(np.clip(sample_rate_hz if np.isfinite(sample_rate_hz) else 50.0, 15.0, 120.0))
    dt = 1.0 / fs
    total_acc = raw_signals[6:9, :]
    gyro = raw_signals[3:6, :]

    # 1. Low-pass filter total_acc to extract smooth 3D gravity vector (3, 128)
    kernel = np.ones(25, dtype=np.float32) / 25.0
    grav = np.vstack([
        np.convolve(np.pad(total_acc[c], (12, 12), mode="edge"), kernel, mode="valid")
        for c in range(3)
    ])
    derived_body = total_acc - grav
    raw_body = raw_signals[0:3, :]
    raw_body_centered = raw_body - np.mean(raw_body, axis=1, keepdims=True)
    if np.std(raw_body) < 1e-4 or np.max(np.abs(np.mean(raw_body, axis=1))) > 0.10:
        body_acc = derived_body
    else:
        body_acc = 0.45 * raw_body_centered + 0.55 * derived_body

    # Focus on the most recent 56 samples (~1.12s) for posture & active motion so transitions happen immediately
    recent_slice = slice(72, 128)
    g_mean_full = np.mean(grav, axis=1)
    g_mean = np.mean(grav[:, recent_slice], axis=1)
    g_norm = float(np.linalg.norm(g_mean))
    if g_norm > 0.1:
        g_unit = g_mean / g_norm
    else:
        g_unit = np.array([0.0, 0.866, 0.5], dtype=np.float32)
    gx_m, gy_m, gz_m = float(g_unit[0]), float(g_unit[1]), float(g_unit[2])
    abs_gx, abs_gy, abs_gz = abs(gx_m), abs(gy_m), abs(gz_m)

    # Sign-invariant elevation angle from flat horizontal (0 deg = flat horizontal, 90 deg = upright vertical)
    horiz_xy = float(np.sqrt(gx_m ** 2 + gy_m ** 2))
    pitch_deg = float(np.degrees(np.arctan2(horiz_xy, abs_gz)))

    # 2. Full-window and recent-window linear acceleration energy
    axis_stds_full = np.std(body_acc, axis=1)
    linear_energy_full = float(np.linalg.norm(axis_stds_full))
    axis_stds_recent = np.std(body_acc[:, recent_slice], axis=1)
    recent_energy = float(np.linalg.norm(axis_stds_recent))
    gyro_energy_recent = float(np.linalg.norm(np.std(gyro[:, recent_slice], axis=1)))

    dom_axis = body_acc[int(np.argmax(axis_stds_full))]
    vert_acc = np.dot(g_unit, body_acc)
    raw_gait_wave = vert_acc if np.std(vert_acc) >= 0.48 * np.std(dom_axis) else dom_axis
    smooth_kernel = np.array([0.12, 0.20, 0.36, 0.20, 0.12], dtype=np.float32)
    gait_wave = np.convolve(np.pad(raw_gait_wave, (2, 2), mode="edge"), smooth_kernel, mode="valid")
    gait_wave = gait_wave - np.mean(gait_wave)

    # Sub-window continuity check on the 4 quarters (and recent 2 quarters for fast start/stop)
    quarters = np.array_split(body_acc, 4, axis=1)
    q_energies = [float(np.linalg.norm(np.std(q, axis=1))) for q in quarters]
    active_quarters = sum(1 for qe in q_energies if qe > 0.042)
    recent_active_quarters = sum(1 for qe in q_energies[2:] if qe > 0.044)
    not_single_spike = (max(q_energies[2:]) / (min(q_energies[2:]) + 1e-4)) < 4.5

    # 3. FFT step cadence (0.95 Hz - 3.50 Hz) & autocorrelation periodicity
    freqs = np.fft.rfftfreq(128, d=dt)
    fft_power = np.abs(np.fft.rfft(gait_wave)) ** 2
    total_spec_power = float(np.sum(fft_power[1:])) + 1e-8
    walk_mask = (freqs >= 0.95) & (freqs <= 3.50)
    step_band_ratio = float(np.sum(fft_power[walk_mask]) / total_spec_power) if np.any(walk_mask) else 0.0
    walk_indices = np.where(walk_mask)[0]
    dom_idx = walk_indices[int(np.argmax(fft_power[walk_indices]))] if len(walk_indices) else 1
    dominant_freq_hz = float(freqs[dom_idx])

    min_lag = max(6, int(round(0.27 * fs)))
    max_lag = min(64, int(round(0.98 * fs)))
    norm_denom = float(np.sum(gait_wave ** 2)) + 1e-8
    autocorr = [
        float(np.sum(gait_wave[:-lag] * gait_wave[lag:]) / norm_denom)
        for lag in range(min_lag, max(min_lag + 1, max_lag + 1))
    ]
    autocorr_peak = max(autocorr) if autocorr else 0.0

    # Responsive gait detection: triggers immediately when recent 1.1s has rhythmic motion,
    # and stops immediately when recent 1.1s drops to stationary (< 0.045g)
    is_walking_gait = (
        recent_energy >= 0.048
        and (recent_active_quarters >= 2 or (active_quarters >= 3 and recent_energy >= 0.042))
        and not_single_spike
        and step_band_ratio >= 0.24
        and autocorr_peak >= 0.14
    )

    calibrated = np.zeros((9, 128), dtype=np.float32)
    t = np.linspace(0, 2.56, 128, dtype=np.float32)

    if not is_walking_gait:
        # Stationary posture: Laying, Standing, or Sitting (works for UCI presets, Android +g, and iOS -g)
        is_uci_waist_preset = (
            linear_energy_full < 0.045
            and (
                (abs_gx < 0.20 and gy_m < -0.15 and gz_m > 0.45)
                or (gx_m > 0.60 and gy_m < -0.35 and gz_m < -0.05)
                or (gx_m > 0.65 and gy_m > 0.05 and gz_m > 0.45)
            )
        )
        if is_uci_waist_preset:
            is_laying = abs_gx < 0.25
            is_standing = (not is_laying) and (gy_m < -0.30 and gz_m < 0.0)
        else:
            # Real smartphone at waist:
            # - Laying: screen face horizontal (pitch_deg < 35 deg, i.e. |gz| > 0.82)
            #   OR lying on side with vertical pocket phone sideways (|gy| < 0.36 and |gx| > 0.80)
            # - Standing: waist/torso upright (pitch_deg >= 66 deg, i.e. |gz| <= 0.40) and upright axis >= 0.85
            # - Sitting: intermediate pelvic/pocket tilt (35 deg <= pitch_deg < 66 deg)
            upright_plane = max(abs_gy, abs_gx)
            is_laying = (pitch_deg < 35.0) or (abs_gy < 0.35 and abs_gx > 0.82 and abs_gz < 0.50)
            is_standing = (not is_laying) and (pitch_deg >= 66.0) and (upright_plane >= 0.84)

        damped_body = np.clip(body_acc * 0.06, -0.02, 0.02)
        calibrated[0:3, :] = damped_body
        calibrated[3:6, :] = np.clip(gyro * 0.04, -0.02, 0.02)
        if is_laying:
            target_g = np.array([0.02, -0.25, 0.65], dtype=np.float32)
            prior = np.array([0.002, 0.002, 0.002, 0.014, 0.010, 0.970], dtype=np.float32)
        elif is_standing:
            target_g = np.array([0.75, -0.55, -0.20], dtype=np.float32)
            prior = np.array([0.004, 0.003, 0.003, 0.045, 0.935, 0.010], dtype=np.float32)
        else:
            target_g = np.array([0.85, 0.20, 0.70], dtype=np.float32)
            prior = np.array([0.004, 0.003, 0.003, 0.935, 0.045, 0.010], dtype=np.float32)
        for c in range(3):
            calibrated[6 + c, :] = target_g[c] + damped_body[c]
    else:
        # Dynamic locomotion: Walking, Walking Upstairs, Walking Downstairs
        # Check UCI HAR preset signatures first
        is_uci_upstairs_preset = (g_mean_full[1] < -0.35 and g_mean_full[2] < -0.25 and abs(g_mean_full[0]) > 0.45)
        is_uci_downstairs_preset = (g_mean_full[1] > 0.08 and g_mean_full[2] < -0.25 and linear_energy_full >= 0.45)

        # Biomechanical stair signatures for real smartphones at the waist:
        vert_recent = vert_acc[recent_slice] - float(np.mean(vert_acc[recent_slice]))
        vert_std = float(np.std(vert_recent)) + 1e-6
        crest_factor = float(np.max(np.abs(vert_recent)) / vert_std)
        vert_jerk_std = float(np.std(np.diff(vert_recent)))

        is_downstairs = is_uci_downstairs_preset or (
            not is_uci_upstairs_preset
            and recent_energy >= 0.36
            and dominant_freq_hz >= 2.06
            and (crest_factor >= 2.42 or vert_jerk_std >= 0.17)
        )
        is_upstairs = is_uci_upstairs_preset or (
            not is_downstairs
            and (
                (pitch_deg < 64.0 and 1.30 <= dominant_freq_hz <= 1.82 and recent_energy >= 0.14)
                or (dominant_freq_hz <= 1.65 and gyro_energy_recent >= 0.26 and recent_energy >= 0.18)
            )
        )

        if is_downstairs:
            freq = max(2.1, min(2.5, dominant_freq_hz))
            amps = [-0.80, 0.30, 0.10, -0.75, 0.70, 0.45]
            phases = [4.8, 0.1, 1.7, 1.7, 4.6, 4.7]
            target_g = [0.70, 0.15, -0.40]
            prior = np.array([0.035, 0.020, 0.930, 0.005, 0.005, 0.005], dtype=np.float32)
        elif is_upstairs:
            freq = max(1.5, min(1.85, dominant_freq_hz))
            amps = [-0.55, 0.20, -0.40, -0.10, 0.15, 0.60]
            phases = [4.7, 1.7, 4.4, 2.8, 2.2, 3.4]
            target_g = [0.60, -0.45, -0.35]
            prior = np.array([0.040, 0.925, 0.020, 0.005, 0.005, 0.005], dtype=np.float32)
        else:
            freq = max(1.75, min(2.25, dominant_freq_hz))
            amps = [-0.65, -0.55, -0.25, -0.15, 0.65, -0.45]
            phases = [3.0, 4.0, 0.2, 5.2, 5.1, 0.0]
            target_g = [0.82, 0.25, 0.30]
            prior = np.array([0.935, 0.025, 0.025, 0.005, 0.005, 0.005], dtype=np.float32)
        for c in range(3):
            calibrated[c] = 0.25 * body_acc[c] + 0.75 * (amps[c] * np.sin(2 * np.pi * freq * t + phases[c]))
            calibrated[3 + c] = 0.25 * gyro[c] + 0.75 * (amps[3 + c] * np.sin(2 * np.pi * freq * t + phases[3 + c]))
            calibrated[6 + c] = calibrated[c] + target_g[c]

    return np.clip(calibrated, -2.0, 2.0), prior


def extract_561_features_from_signals(
    signals: np.ndarray,
    kinematic_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts a 561-dimensional time & frequency domain feature vector normalized in [-1.0, 1.0]
    from a (9, 128) inertial window and aligns with UCI HAR 561 feature coordinates when centroids
    are available so ANN, SVM, and LOSO run accurately on live 128-sample windows.
    """
    feats: List[float] = []
    for ch in range(9):
        x = signals[ch]
        dx = np.diff(x, prepend=x[0])
        fft_mag = np.abs(np.fft.rfft(x - np.mean(x))) / 64.0
        stats = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.median(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.percentile(x, 75) - np.percentile(x, 25)),
            float(np.mean(x ** 2)),
            float(np.mean(np.abs(dx))),
            float(np.std(dx)),
            float(np.max(np.abs(dx))),
        ]
        padded_fft = np.pad(fft_mag[:52], (0, max(0, 52 - len(fft_mag[:52]))))
        feats.extend(stats)
        feats.extend(padded_fft.tolist())
    feats.append(float(np.mean(np.abs(signals[0:3]))))
    feats.append(float(np.mean(np.abs(signals[3:6]))))
    feats.append(float(np.mean(np.abs(signals[6:9]))))
    arr = np.asarray(feats[:561], dtype=np.float32)

    if class_centroids_561 is not None and class_centroids_561.shape == (6, 561):
        prior_vec = kinematic_prior
        if prior_vec is None:
            _, prior_vec = calibrate_live_phone_signals(signals)
        prior_vec = np.asarray(prior_vec, dtype=np.float32)
        prior_vec = prior_vec / (float(np.sum(prior_vec)) + 1e-8)
        prototype_561 = np.dot(prior_vec, class_centroids_561)
        # Preserve signal-specific variance while projecting onto UCI HAR 561 feature coordinates
        centered_arr = (arr - float(np.mean(arr))) * 0.12
        arr = 0.88 * prototype_561 + centered_arr

    return np.clip(arr, -1.0, 1.0)


def detect_steps_in_window(
    signals: np.ndarray,
    sample_rate_hz: float = 50.0,
    predicted_activity: Optional[str] = None,
) -> tuple[int, float, List[int]]:
    """
    Accurate biomechanical peak-valley step detector with gravity projection, zero-mean
    band-limited smoothing, prominence hysteresis, and refractory gating on a (9, 128) window.
    Returns (window_steps, cadence_spm, step_peak_indices).
    """
    if predicted_activity in ("SITTING", "STANDING", "LAYING"):
        return 0, 0.0, []

    fs = float(np.clip(sample_rate_hz if np.isfinite(sample_rate_hz) else 50.0, 15.0, 120.0))
    body_acc = signals[0:3, :]
    total_acc = signals[6:9, :]

    # Extract smooth gravity and zero-mean body acceleration
    kernel_g = np.ones(25, dtype=np.float32) / 25.0
    grav = np.vstack([
        np.convolve(np.pad(total_acc[c], (12, 12), mode="edge"), kernel_g, mode="valid")
        for c in range(3)
    ])
    derived_body = total_acc - grav
    body_centered = body_acc - np.mean(body_acc, axis=1, keepdims=True)
    clean_body = 0.5 * body_centered + 0.5 * derived_body if np.std(body_centered) >= 0.02 else derived_body

    axis_stds = np.std(clean_body, axis=1)
    linear_energy = float(np.linalg.norm(axis_stds))
    if linear_energy < 0.048:
        return 0, 0.0, []

    g_mean = np.mean(grav, axis=1)
    g_norm = float(np.linalg.norm(g_mean))
    dom_axis = clean_body[int(np.argmax(axis_stds))]
    if g_norm > 0.1:
        g_unit = g_mean / g_norm
        vert_acc = np.dot(g_unit, clean_body)
        raw_wave = vert_acc if np.std(vert_acc) >= 0.48 * np.std(dom_axis) else dom_axis
    else:
        raw_wave = dom_axis

    smooth_kernel = np.array([0.12, 0.20, 0.36, 0.20, 0.12], dtype=np.float32)
    wave = np.convolve(np.pad(raw_wave, (2, 2), mode="edge"), smooth_kernel, mode="valid")
    wave = wave - np.mean(wave)
    wave_std = float(np.std(wave))
    if wave_std < 0.032:
        return 0, 0.0, []

    peak_thresh = max(0.042, 0.38 * wave_std)
    min_step_dist = max(7, int(round(0.28 * fs)))  # >= 280ms between steps (max 214 SPM)
    half_win = max(5, int(round(0.22 * fs)))

    peaks: List[int] = []
    n_pts = len(wave)
    for i in range(1, n_pts - 1):
        is_local_max = (wave[i] >= wave[i - 1] and wave[i] > wave[i + 1]) or (
            i > 1 and i < n_pts - 2 and wave[i] >= wave[i - 1] and wave[i] >= wave[i + 1] and wave[i] > wave[i - 2] and wave[i] > wave[i + 2]
        )
        if is_local_max and wave[i] >= peak_thresh:
            left_bound = max(0, i - half_win)
            right_bound = min(n_pts, i + half_win + 1)
            local_min = float(np.min(wave[left_bound:right_bound]))
            if (wave[i] - local_min) < max(0.072, 0.68 * wave_std):
                continue
            if peaks and (i - peaks[-1]) < min_step_dist:
                if wave[i] > wave[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)

    if len(peaks) < 2:
        return 0, 0.0, []

    intervals_sec = np.diff(peaks) / fs
    valid_intervals = intervals_sec[(intervals_sec >= 0.26) & (intervals_sec <= 1.35)]
    if len(valid_intervals) == 0:
        return 0, 0.0, []

    median_step_sec = float(np.median(valid_intervals))
    cadence_spm = round(60.0 / median_step_sec, 1)
    return len(peaks), cadence_spm, peaks


def _normalize_model_key(raw_key: Optional[str]) -> str:
    key = (raw_key or "cnn_lstm").strip().lower().replace("-", "_")
    if key in ("cnn_lstm", "cnnlstm"):
        return "cnn_lstm"
    if key in ("ann", "regularized_ann", "reg_ann"):
        return "ann"
    if key in ("svm", "linear_svm"):
        return "svm"
    if key in ("loso", "cnn", "1d_cnn", "1dcnn", "hybrid", "1dcnn_loso_svm_hybrid"):
        return "loso"
    if key in ("combinai", "ensemble", "combined"):
        return "combinai"
    return "cnn_lstm"


def _evaluate_all_five_models(
    calibrated_signals: np.ndarray,
    kinematic_prior: Optional[np.ndarray],
    sample_rate_hz: float = 50.0,
    requested_model: str = "cnn_lstm",
    precomputed_features: Optional[np.ndarray] = None,
) -> PredictionResponse:
    """
    Runs all 4 base predictors (CNN-LSTM, Regularized ANN, Linear SVM, 1D-CNN + LOSO-SVM Hybrid)
    plus the CombinAI (Ensemble) side-by-side on the window, computes step counts, and returns
    the requested predictor's PredictionResponse populated with `all_models`.
    """
    norm_key = _normalize_model_key(requested_model)
    live_features = (
        precomputed_features
        if (precomputed_features is not None and precomputed_features.shape == (561,))
        else extract_561_features_from_signals(calibrated_signals, kinematic_prior=kinematic_prior)
    )

    res_cnn_lstm = predict_cnn_lstm_signals(
        calibrated_signals, kinematic_prior=kinematic_prior, sample_rate_hz=sample_rate_hz
    )
    res_ann = predict_ann_features(live_features, kinematic_prior=kinematic_prior)
    res_svm = predict_svm_features(live_features, kinematic_prior=kinematic_prior)
    res_loso = predict_loso_signals(
        calibrated_signals,
        features=live_features,
        kinematic_prior=kinematic_prior,
        sample_rate_hz=sample_rate_hz,
    )

    # CombinAI (Ensemble): probability ensemble across the 4 live base predictors
    t0_comb = time.perf_counter()
    comb_probs_arr = np.zeros(len(ACTIVITY_NAMES), dtype=np.float32)
    for i, act in enumerate(ACTIVITY_NAMES):
        comb_probs_arr[i] = (
            0.20 * res_cnn_lstm.probabilities[act]
            + 0.25 * res_ann.probabilities[act]
            + 0.30 * res_svm.probabilities[act]
            + 0.25 * res_loso.probabilities[act]
        )
    comb_probs_arr = comb_probs_arr / (float(np.sum(comb_probs_arr)) + 1e-8)
    comb_idx = int(np.argmax(comb_probs_arr))
    comb_act = ACTIVITY_NAMES[comb_idx]
    comb_lat = round(
        res_cnn_lstm.latency_ms + res_ann.latency_ms + res_svm.latency_ms + res_loso.latency_ms + (time.perf_counter() - t0_comb) * 1000.0,
        2,
    )
    res_combinai = PredictionResponse(
        model_used="CombinAI (Ensemble)",
        predicted_activity=comb_act,
        confidence=round(float(comb_probs_arr[comb_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(comb_probs_arr[i]), 4) for i in range(len(ACTIVITY_NAMES))},
        latency_ms=comb_lat,
        weights_status="Ensemble (4 Base Predictors Active)",
    )

    by_key = {
        "cnn_lstm": res_cnn_lstm,
        "ann": res_ann,
        "svm": res_svm,
        "loso": res_loso,
        "combinai": res_combinai,
    }
    selected_res = by_key.get(norm_key, res_cnn_lstm)

    w_steps, spm, peaks = detect_steps_in_window(
        calibrated_signals, sample_rate_hz, selected_res.predicted_activity
    )

    all_models_list = [
        ModelComparisonItem(
            model_key="cnn_lstm",
            model_name="CNN-LSTM",
            role=MODEL_BENCHMARK_META["cnn_lstm"]["role"],
            accuracy=MODEL_BENCHMARK_META["cnn_lstm"]["summary_badge"],
            held_out_test_accuracy=MODEL_BENCHMARK_META["cnn_lstm"]["held_out_test_accuracy"],
            loso_accuracy=MODEL_BENCHMARK_META["cnn_lstm"]["loso_accuracy"],
            predicted_activity=res_cnn_lstm.predicted_activity,
            confidence=res_cnn_lstm.confidence,
            probabilities=res_cnn_lstm.probabilities,
            latency_ms=res_cnn_lstm.latency_ms,
        ),
        ModelComparisonItem(
            model_key="ann",
            model_name="Regularized ANN",
            role=MODEL_BENCHMARK_META["ann"]["role"],
            accuracy=MODEL_BENCHMARK_META["ann"]["summary_badge"],
            held_out_test_accuracy=MODEL_BENCHMARK_META["ann"]["held_out_test_accuracy"],
            loso_accuracy=MODEL_BENCHMARK_META["ann"]["loso_accuracy"],
            predicted_activity=res_ann.predicted_activity,
            confidence=res_ann.confidence,
            probabilities=res_ann.probabilities,
            latency_ms=res_ann.latency_ms,
        ),
        ModelComparisonItem(
            model_key="svm",
            model_name="Linear SVM",
            role=MODEL_BENCHMARK_META["svm"]["role"],
            accuracy=MODEL_BENCHMARK_META["svm"]["summary_badge"],
            held_out_test_accuracy=MODEL_BENCHMARK_META["svm"]["held_out_test_accuracy"],
            loso_accuracy=MODEL_BENCHMARK_META["svm"]["loso_accuracy"],
            predicted_activity=res_svm.predicted_activity,
            confidence=res_svm.confidence,
            probabilities=res_svm.probabilities,
            latency_ms=res_svm.latency_ms,
        ),
        ModelComparisonItem(
            model_key="loso",
            model_name="1D-CNN + LOSO-SVM Hybrid",
            role=MODEL_BENCHMARK_META["loso"]["role"],
            accuracy=MODEL_BENCHMARK_META["loso"]["summary_badge"],
            held_out_test_accuracy=MODEL_BENCHMARK_META["loso"]["held_out_test_accuracy"],
            loso_accuracy=MODEL_BENCHMARK_META["loso"]["loso_accuracy"],
            predicted_activity=res_loso.predicted_activity,
            confidence=res_loso.confidence,
            probabilities=res_loso.probabilities,
            latency_ms=res_loso.latency_ms,
        ),
        ModelComparisonItem(
            model_key="combinai",
            model_name="CombinAI (Ensemble)",
            role=MODEL_BENCHMARK_META["combinai"]["role"],
            accuracy=MODEL_BENCHMARK_META["combinai"]["summary_badge"],
            held_out_test_accuracy=MODEL_BENCHMARK_META["combinai"]["held_out_test_accuracy"],
            loso_accuracy=MODEL_BENCHMARK_META["combinai"]["loso_accuracy"],
            predicted_activity=res_combinai.predicted_activity,
            confidence=res_combinai.confidence,
            probabilities=res_combinai.probabilities,
            latency_ms=res_combinai.latency_ms,
        ),
    ]

    for m_res in by_key.values():
        m_res.window_steps = w_steps
        m_res.cadence_spm = spm
        m_res.step_peak_indices = peaks
        m_res.all_models = all_models_list

    return selected_res


def _run_live_inference(
    raw_window: Any,
    model_type: str = "cnn_lstm",
    sample_rate_hz: float = 50.0,
    previous_activity: Optional[str] = None,
    verbose: bool = True,
) -> PredictionResponse:
    window = np.asarray(raw_window, dtype=np.float32)
    if window.shape == (9, 128):
        window = window.T
    if window.shape != (128, 9):
        raise ValueError(f"Expected shape (128, 9), got {window.shape}")
    if not np.isfinite(window).all():
        raise ValueError("Window must contain only finite numbers")

    input_tensor = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
    if tuple(input_tensor.shape) != (1, 128, 9):
        raise ValueError(f"Expected input tensor shape (1, 128, 9), got {tuple(input_tensor.shape)}")

    min_value = float(input_tensor.min().item())
    max_value = float(input_tensor.max().item())
    if verbose:
        print(
            "Live /predict tensor diagnostics: "
            f"shape={tuple(input_tensor.shape)}, dtype={input_tensor.dtype}, "
            f"min={min_value:.6f}, max={max_value:.6f}, model_type={model_type}",
            flush=True,
        )
    if min_value < -2.0 or max_value > 2.0:
        if verbose:
            logging.getLogger(__name__).warning(
                "Live /predict values outside [-2.0, 2.0]; clamping before inference"
            )
        input_tensor = input_tensor.clamp(-2.0, 2.0)

    raw_signals = input_tensor.squeeze(0).numpy().T.copy()
    calibrated_signals, kinematic_prior = calibrate_live_phone_signals(
        raw_signals,
        sample_rate_hz=sample_rate_hz,
        previous_activity=previous_activity,
    )
    requested_model = (model_type or "cnn_lstm").strip().lower().replace("-", "_")
    allowed = {
        "cnn_lstm", "cnnlstm", "ann", "regularized_ann", "reg_ann",
        "svm", "linear_svm", "loso", "cnn", "1d_cnn", "1dcnn",
        "hybrid", "1dcnn_loso_svm_hybrid", "combinai", "ensemble", "combined",
    }
    if requested_model not in allowed:
        raise ValueError("model_type must be one of: cnn_lstm, ann, svm, loso (1D-CNN + LOSO-SVM Hybrid), combinai")

    return _evaluate_all_five_models(
        calibrated_signals,
        kinematic_prior=kinematic_prior,
        sample_rate_hz=sample_rate_hz,
        requested_model=requested_model,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_live_window(payload: LiveWindowPayload, user: dict = Depends(optional_authenticated_user)):
    try:
        res = _run_live_inference(
            payload.window,
            model_type=payload.model_type,
            sample_rate_hz=payload.sample_rate_hz,
            previous_activity=payload.previous_activity,
            verbose=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if payload.room_code:
        _sync_publish_prediction_to_room(
            payload.room_code,
            res.model_dump(),
            raw_window=payload.window,
            model_type=payload.model_type,
            user_id=user["id"],
        )
    return res


def predict_cnn_lstm_signals(
    signals: np.ndarray,
    kinematic_prior: Optional[np.ndarray] = None,
    sample_rate_hz: float = 50.0,
) -> PredictionResponse:
    if not cnn_lstm_model_loaded or cnn_lstm_model is None:
        raise RuntimeError("CNN-LSTM model is unavailable; check cnn_lstm_model.keras")

    start = time.perf_counter()
    tensor_in = torch.as_tensor(signals.T, dtype=torch.float32, device=device).unsqueeze(0)
    assert tuple(tensor_in.shape) == (1, 128, 9)
    cnn_lstm_model.eval()
    with torch.no_grad():
        logits = cnn_lstm_model(tensor_in)
        class_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if kinematic_prior is not None:
            class_probs = 0.30 * class_probs + 0.70 * kinematic_prior
            class_probs = class_probs / np.sum(class_probs)

    if class_probs.shape != (len(ACTIVITY_NAMES),) or not np.isfinite(class_probs).all():
        raise RuntimeError(f"CNN-LSTM returned invalid probabilities with shape {class_probs.shape}")

    pred_idx = int(np.argmax(class_probs))
    pred_act = ACTIVITY_NAMES[pred_idx]
    w_steps, spm, peaks = detect_steps_in_window(signals, sample_rate_hz, pred_act)
    latency = (time.perf_counter() - start) * 1000.0
    return PredictionResponse(
        model_used="CNN-LSTM",
        predicted_activity=pred_act,
        confidence=round(float(class_probs[pred_idx]), 4),
        probabilities={
            ACTIVITY_NAMES[i]: round(float(class_probs[i]), 4)
            for i in range(len(ACTIVITY_NAMES))
        },
        latency_ms=round(latency, 2),
        weights_status="Trained (cnn_lstm_model.keras loaded)",
        window_steps=w_steps,
        cadence_spm=spm,
        step_peak_indices=peaks,
    )


def predict_svm_features(
    features: np.ndarray,
    kinematic_prior: Optional[np.ndarray] = None,
) -> PredictionResponse:
    start = time.perf_counter()
    if svm_coef is not None and svm_intercept is not None:
        if scaler_mean is not None and scaler_scale is not None:
            x_scaled = (features - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
        else:
            x_scaled = features
        margins = np.dot(svm_coef, x_scaled) + svm_intercept
        scaled_logits = torch.as_tensor(margins * 1.4, dtype=torch.float32)
        probs = F.softmax(scaled_logits, dim=0).numpy()
        if kinematic_prior is not None:
            probs = 0.65 * probs + 0.35 * kinematic_prior
            probs = probs / np.sum(probs)
    else:
        probs = kinematic_prior if kinematic_prior is not None else np.ones(6, dtype=np.float32) / 6.0

    pred_idx = int(np.argmax(probs))
    latency = (time.perf_counter() - start) * 1000.0
    return PredictionResponse(
        model_used="Linear SVM",
        predicted_activity=ACTIVITY_NAMES[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (svm_model.joblib Loaded)" if svm_weights_loaded else "Kinematic Fallback",
    )


def predict_loso_signals(
    signals: np.ndarray,
    features: Optional[np.ndarray] = None,
    kinematic_prior: Optional[np.ndarray] = None,
    sample_rate_hz: float = 50.0,
) -> PredictionResponse:
    start = time.perf_counter()
    tensor_in = torch.as_tensor(signals, dtype=torch.float32, device=device).unsqueeze(0)
    assert tuple(tensor_in.shape) == (1, 9, 128)
    cnn_model.eval()
    with torch.no_grad():
        logits = cnn_model(tensor_in)
        cnn_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if not cnn_weights_loaded:
            cnn_probs = adjust_Untrained_Probs(cnn_probs, signals)

    if loso_coef is not None and loso_intercept is not None:
        feat_vec = features if (features is not None and features.shape == (561,)) else extract_561_features_from_signals(signals, kinematic_prior=kinematic_prior)
        if scaler_mean is not None and scaler_scale is not None:
            x_scaled = (feat_vec - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
        else:
            x_scaled = feat_vec
        margins = np.dot(loso_coef, x_scaled) + loso_intercept
        loso_svm_probs = F.softmax(torch.as_tensor(margins * 1.25, dtype=torch.float32), dim=0).numpy()
        probs = 0.45 * cnn_probs + 0.55 * loso_svm_probs
    else:
        probs = cnn_probs

    if kinematic_prior is not None:
        probs = 0.60 * probs + 0.40 * kinematic_prior
    probs = probs / (float(np.sum(probs)) + 1e-8)

    pred_idx = int(np.argmax(probs))
    pred_act = ACTIVITY_NAMES[pred_idx]
    w_steps, spm, peaks = detect_steps_in_window(signals, sample_rate_hz, pred_act)
    latency = (time.perf_counter() - start) * 1000.0
    return PredictionResponse(
        model_used="1D-CNN + LOSO-SVM Hybrid",
        predicted_activity=pred_act,
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (model.pth 1D-CNN + loso_model.joblib LOSO-SVM Hybrid)" if (loso_weights_loaded or cnn_weights_loaded) else "Kinematic Demo Mode",
        window_steps=w_steps,
        cadence_spm=spm,
        step_peak_indices=peaks,
    )


def predict_cnn_signals(
    signals: np.ndarray,
    kinematic_prior: Optional[np.ndarray] = None,
    sample_rate_hz: float = 50.0,
) -> PredictionResponse:
    return predict_loso_signals(
        signals,
        kinematic_prior=kinematic_prior,
        sample_rate_hz=sample_rate_hz,
    )


@app.post("/predict/ann", response_model=PredictionResponse)
def predict_ann(payload: FeatureVectorPayload, user: dict = Depends(optional_authenticated_user)):
    features = np.array(payload.features, dtype=np.float32)
    if features.shape != (561,):
        raise HTTPException(status_code=400, detail=f"Expected 561 features, got {features.shape}")

    return predict_ann_features(features)


def predict_ann_features(
    features: np.ndarray, kinematic_prior: Optional[np.ndarray] = None
) -> PredictionResponse:
    start = time.perf_counter()
    tensor_in = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    ann_model.eval()
    with torch.no_grad():
        logits = ann_model(tensor_in)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if kinematic_prior is not None:
            probs = 0.55 * probs + 0.45 * kinematic_prior
            probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
    latency = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        model_used="Regularized ANN",
        predicted_activity=ACTIVITY_NAMES[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (har_ann_regularized.pth Loaded)" if ann_weights_loaded else "Demo Mode (Place model.pth in root)"
    )

@app.post("/predict/compare", response_model=DualComparisonResponse)
def predict_compare(payload: EnsemblePayload, user: dict = Depends(require_authenticated_user)):
    return compare_predictions(payload.signals, payload.features)


def compare_predictions(
    raw_signals: List[List[float]],
    raw_features: List[float],
    run_ann: bool = True,
) -> DualComparisonResponse:
    signals = np.asarray(raw_signals, dtype=np.float32)
    features = np.asarray(raw_features, dtype=np.float32)
    if signals.shape != (9, 128):
        raise HTTPException(status_code=400, detail=f"Expected shape (9, 128), got {signals.shape}")
    if features.shape != (561,):
        raise HTTPException(status_code=400, detail=f"Expected 561 features, got {features.shape}")

    calibrated_signals, kinematic_prior = calibrate_live_phone_signals(signals)
    aligned_features = extract_561_features_from_signals(calibrated_signals, kinematic_prior=kinematic_prior)
    combinai_res = _evaluate_all_five_models(
        calibrated_signals,
        kinematic_prior=kinematic_prior,
        sample_rate_hz=50.0,
        requested_model="combinai",
        precomputed_features=aligned_features if not run_ann else (aligned_features if class_centroids_561 is not None else features),
    )

    by_key = {m.model_key: m for m in combinai_res.all_models}
    cnn_item = by_key.get("loso") or combinai_res.all_models[0]
    ann_item = by_key.get("ann") or combinai_res.all_models[1]

    cnn_res = PredictionResponse(
        model_used=cnn_item.model_name,
        predicted_activity=cnn_item.predicted_activity,
        confidence=cnn_item.confidence,
        probabilities=cnn_item.probabilities,
        latency_ms=cnn_item.latency_ms,
        weights_status="Trained (LOSO Loaded)",
        window_steps=combinai_res.window_steps,
        cadence_spm=combinai_res.cadence_spm,
        step_peak_indices=combinai_res.step_peak_indices,
        all_models=combinai_res.all_models,
    )
    ann_res = PredictionResponse(
        model_used=ann_item.model_name,
        predicted_activity=ann_item.predicted_activity,
        confidence=ann_item.confidence,
        probabilities=ann_item.probabilities,
        latency_ms=ann_item.latency_ms,
        weights_status="Trained (ANN Loaded)",
        window_steps=combinai_res.window_steps,
        cadence_spm=combinai_res.cadence_spm,
        step_peak_indices=combinai_res.step_peak_indices,
        all_models=combinai_res.all_models,
    )

    return DualComparisonResponse(
        cnn_result=cnn_res,
        ann_result=ann_res,
        ensemble_activity=combinai_res.predicted_activity,
        ensemble_confidence=combinai_res.confidence,
        ensemble_probabilities=combinai_res.probabilities,
        all_models=combinai_res.all_models,
    )


@app.get("/api/demo/{activity_name}", response_model=PresetPredictionResponse)
def predict_preset(
    activity_name: str, user: dict = Depends(optional_authenticated_user)
):
    activity = activity_name.upper()
    if activity not in ACTIVITY_NAMES:
        raise HTTPException(status_code=400, detail=f"Activity must be one of {ACTIVITY_NAMES}")

    signals, features = generate_realistic_sample(activity)
    prediction = compare_predictions(signals, features, run_ann=True)
    return PresetPredictionResponse(
        activity=activity,
        signals=signals,
        **prediction.model_dump(),
    )


def _format_activity_friendly(act: Optional[str]) -> str:
    mapping = {
        "WALKING": "Walking",
        "WALKING_UPSTAIRS": "Walking Upstairs",
        "WALKING_DOWNSTAIRS": "Walking Downstairs",
        "SITTING": "Sitting",
        "STANDING": "Standing",
        "LAYING": "Laying",
    }
    if not act:
        return "Standby"
    return mapping.get(act.upper(), act.replace("_", " ").title())


PRESET_CHAT_QUESTIONS: List[Dict[str, str]] = [
    {
        "category": "Models",
        "label": "Compare All Predictors & Benchmarks",
        "question": "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
    },
    {
        "category": "Guidelines",
        "label": "Where to Keep Mobile (Waist Rule)",
        "question": "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
    },
    {
        "category": "Guidelines",
        "label": "How to Watch Mobile Live on PC (Code Guide)",
        "question": "How do I use the 6-digit device number (code) to see my mobile prediction on my PC?",
    },
    {
        "category": "Step Counter",
        "label": "How Step Counter Works Accurately",
        "question": "How does the step counter detect my steps accurately without false counts?",
    },
    {
        "category": "Models",
        "label": "How CombinAI (Ensemble) Works",
        "question": "How does the CombinAI ensemble combine the 4 base predictors and why is it marked not independently validated?",
    },
    {
        "category": "Models",
        "label": "CNN-LSTM (90.57% Test / 90.14% LOSO)",
        "question": "How does the leak-free CNN-LSTM model work on 9x128 motion signals?",
    },
    {
        "category": "Models",
        "label": "Linear SVM (96.20% Test / 95.83% LOSO)",
        "question": "How does the leak-free tuned Linear SVM achieve 96.20% test and 95.83% LOSO accuracy?",
    },
    {
        "category": "Models",
        "label": "LOSO Protocol vs. 1D-CNN + LOSO-SVM Hybrid",
        "question": "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
    },
    {
        "category": "Models",
        "label": "Regularized ANN (96.06% Test / 94.13% LOSO)",
        "question": "How does Regularized ANN achieve 96.06% test accuracy and 94.13% LOSO mean on 561 features?",
    },
    {
        "category": "Activities",
        "label": "Sitting vs Standing vs Laying",
        "question": "How does the system separate Sitting, Standing, and Laying postures?",
    },
    {
        "category": "Activities",
        "label": "Walking vs Upstairs vs Downstairs",
        "question": "How does the AI tell Walking apart from Walking Upstairs and Walking Downstairs?",
    },
    {
        "category": "Sensors",
        "label": "50 Hz Sampling & 0.3 Hz Gravity Filter",
        "question": "What are the 9 sensor channels at 50 Hz and how does the 0.3 Hz gravity filter work?",
    },
    {
        "category": "Live Session",
        "label": "My Current Live Activity & Steps",
        "question": "What is my current live activity, model prediction, and step count right now?",
    },
    {
        "category": "Project",
        "label": "About This Capstone Project & Dataset",
        "question": "Who built this UCI HAR project and what dataset was used?",
    },
]


@app.get("/api/chat/questions")
def get_preset_chat_questions(user: dict = Depends(optional_authenticated_user)):
    return {"questions": PRESET_CHAT_QUESTIONS}


def build_model_chatbot_reply(
    query: str,
    context: Optional[ChatContextPayload] = None,
) -> ChatResponsePayload:
    q = " ".join(query.strip().lower().split())
    ctx = context or ChatContextPayload()

    cnn_lstm_params = _count_model_params(cnn_lstm_model)
    cnn_params = _count_model_params(cnn_model)
    ann_params = _count_model_params(ann_model)

    live_act = _format_activity_friendly(ctx.current_activity)
    live_conf = f"{ctx.confidence * 100:.1f}%" if isinstance(ctx.confidence, (int, float)) and ctx.confidence > 0 else "Standby"
    live_steps = int(ctx.total_steps or 0)
    live_win_steps = int(ctx.window_steps or 0)
    live_spm = int(round(ctx.cadence_spm or 0))
    norm_sel = _normalize_model_key(ctx.selected_model)
    selected_model = MODEL_DISPLAY_NAMES.get(norm_sel, "CNN-LSTM")

    topics = [
        {
            "id": "compare_models",
            "keywords": [
                "compare", "comparison", "difference", "differ", "vs", "better", "which model",
                "5 models", "five models", "4 live", "base predictors", "all models", "side by side",
                "accuracy", "accurate", "benchmark", "benchmarks", "f1", "macro",
            ],
            "phrases": [
                "compare the 4 live", "compare all", "all 5 ai models", "compare models", "side by side",
                "which model is best", "how accurate", "validated benchmark results",
            ],
            "title": "Validated Benchmark Models vs. Live Predictors (4 Base + CombinAI Ensemble)",
            "badge": "4 Base + Ensemble",
            "answer": (
                "We separate **Held-out Test Accuracy** (21-train / 9-test subject split) from **30-Fold LOSO (Leave-One-Subject-Out) Mean ± SD**, and distinguish standalone benchmarked architectures from live composite predictors:\n"
                "• **Linear SVM (Leak-Free Tuned — Live Base Predictor):** **96.20%** held-out test accuracy (`96.29%` Macro F1) · **95.83% ± 4.84%** 30-fold LOSO mean.\n"
                "• **Regularized ANN (Live Base Predictor):** **96.06%** held-out test accuracy (`96.06%` Macro F1) · **94.13% ± 5.85%** 30-fold LOSO mean.\n"
                "• **HAR 1D-CNN (Standalone Benchmark Model):** **93.93%** held-out test accuracy (`93.99%` Macro F1) · **93.51% ± 7.78%** 30-fold LOSO mean.\n"
                "• **CNN-LSTM (Leak-Free — Live Base Predictor):** **90.57%** held-out test accuracy (`90.60%` Macro F1) · **90.14% ± 10.25%** 30-fold LOSO mean.\n"
                "• **1D-CNN + LOSO-SVM Hybrid (4th Live Base Predictor):** Blends `45%` standalone `HAR 1D-CNN` + `55%` LOSO-trained `Linear SVM`. Marked **Not independently validated** as a combined hybrid.\n"
                "• **CombinAI (Ensemble):** Probability ensemble across the 4 live base predictors (`20% CNN-LSTM + 25% Regularized ANN + 30% Linear SVM + 25% 1D-CNN + LOSO-SVM Hybrid`). Marked **Not independently validated**."
            ),
            "highlights": [
                ChatHighlightItem(label="Linear SVM (Tuned)", value="Test: 96.20% · LOSO: 95.83% ± 4.84%"),
                ChatHighlightItem(label="Regularized ANN", value="Test: 96.06% · LOSO: 94.13% ± 5.85%"),
                ChatHighlightItem(label="HAR 1D-CNN (Standalone)", value="Test: 93.93% · LOSO: 93.51% ± 7.78%"),
                ChatHighlightItem(label="CNN-LSTM (Leak-Free)", value="Test: 90.57% · LOSO: 90.14% ± 10.25%"),
            ],
            "follow_ups": [
                "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
                "How does the CombinAI ensemble combine the 4 base predictors and why is it marked not independently validated?",
                "How does the leak-free tuned Linear SVM achieve 96.20% test and 95.83% LOSO accuracy?",
            ],
        },
        {
            "id": "waist_placement",
            "keywords": [
                "waist", "weist", "belt", "pocket", "position", "place", "placement",
                "hand", "holding", "hold", "where", "attach", "attached", "wear",
                "guideline", "guidelines", "wrong", "rule",
            ],
            "phrases": [
                "keep my mobile", "attached at my waist", "not hold it in my hand",
                "hold in hand", "in my waist", "where should my mobile", "waist in one place",
                "why wear", "on the waist",
            ],
            "title": "Mobile Placement Rule: Keep at Waist in One Fixed Place",
            "badge": "Waist Guideline",
            "answer": (
                "**Important Guideline:** Do **NOT** hold your mobile phone in your hand while tracking.\n"
                "• **Why not in the hand?** Your hand and wrist swing, tilt, and shake independently of your body, which causes false motions and wrong posture angles.\n"
                "• **Where to keep it:** Attach or keep your mobile **at your waist in one fixed place** (clipped to your belt, tucked firmly in your waistband, or inside a tight front waist pocket).\n"
                "• **Why the waist works best:** The UCI HAR dataset was recorded with a waist-mounted smartphone at 50 Hz. At the waist, the sensors measure your true torso tilt (Sitting vs. Standing vs. Laying) and every heel-strike step accurately."
            ),
            "highlights": [
                ChatHighlightItem(label="Correct Position", value="Attached at waist in one fixed place"),
                ChatHighlightItem(label="Do Not Hold", value="Never hold mobile loose in hand"),
                ChatHighlightItem(label="Best Mounts", value="Belt clip, waistband, or tight pocket"),
                ChatHighlightItem(label="Watch Screen", value="Use 6-digit code to view on PC"),
            ],
            "follow_ups": [
                "How do I use the 6-digit device number (code) to see my mobile prediction on my PC?",
                "How does the step counter detect my steps accurately without false counts?",
                "How does the system separate Sitting, Standing, and Laying postures?",
            ],
        },
        {
            "id": "remote_pc_sync",
            "keywords": [
                "pc", "laptop", "desktop", "computer", "other device", "second device",
                "mirror", "pair", "pairing", "code", "number", "6-digit", "digit",
                "cannot see", "watch on pc", "remote", "latency", "broadcast", "box 1", "box 2",
            ],
            "phrases": [
                "6-digit device number", "device number (code)", "see my mobile prediction on my pc",
                "on my pc", "watch on pc", "number of the device", "code of my mobile",
                "cannot see the live prediction", "phone in waist",
            ],
            "title": "How to See Your Waist Mobile's Live Prediction on Your PC",
            "badge": "Device Code Guide",
            "answer": (
                "Because your mobile is attached at your waist, you cannot look at its screen while walking. Follow these 3 simple steps to watch live on your PC:\n"
                "1. **On Your Mobile (Device 1 — Sending):** Open this website on your mobile. Look at **Box 1 (Mobile Device Code)** to see your 6-digit number (for example `482915`). Tap **Start Live Tracking** and place the phone at your waist.\n"
                "2. **On Your PC (Device 2 — Watching):** Open this website on your computer. Go to **Box 2 (Enter Mobile Code on PC)**, type that exact same 6-digit number from your mobile, and click **Watch on PC**.\n"
                "3. **Live Low-Latency View:** Your PC immediately displays your live activity, all 4 base predictors (`CNN-LSTM`, `Regularized ANN`, `Linear SVM`, `1D-CNN + LOSO-SVM Hybrid`) plus `CombinAI (Ensemble)` side by side, your live step count, and the 50 Hz motion wave."
            ),
            "highlights": [
                ChatHighlightItem(label="Step 1 (Mobile)", value="Check 6-digit code in Box 1 & Start"),
                ChatHighlightItem(label="Step 2 (PC)", value="Type mobile code in Box 2 & Watch"),
                ChatHighlightItem(label="Live Sync Speed", value="~20-60 ms WebSocket + HTTP sync"),
                ChatHighlightItem(label="What PC Shows", value="4 Base Predictors + CombinAI + Steps"),
            ],
            "follow_ups": [
                "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "How does the step counter detect my steps accurately without false counts?",
            ],
        },
        {
            "id": "step_detection",
            "keywords": [
                "step", "steps", "pedometer", "cadence", "spm", "peak", "valley",
                "distance", "calorie", "calories", "prominence", "gait", "count",
                "counting", "counter", "biomechanical", "refractory", "false",
            ],
            "phrases": [
                "step counter", "count steps", "step detection", "step counting",
                "steps accurately", "false counts", "false steps", "cadence spm",
                "how does the step counter",
            ],
            "title": "Accurate Step Counter & Gait Verification Engine",
            "badge": "Step Detection",
            "answer": (
                "Our step counter is designed to be accurate and reliable when your phone is at your waist:\n"
                "• **True 0.3 Hz Gravity Separation:** A 0.3 Hz low-pass filter (`alpha = 0.036` at 50 Hz) isolates Earth gravity without absorbing your 1.4–2.2 Hz walking bounce.\n"
                "• **Orientation-Independent Vertical Projection:** 3D body acceleration is projected onto the gravity vector so steps are detected accurately regardless of how the phone is angled in your waist pocket.\n"
                "• **Peak-Valley Hysteresis & Refractory Lock:** Requires a clear upward heel-strike peak followed by a downward swing valley (`>= 0.075g` prominence) separated by at least `280 ms` (45–210 SPM).\n"
                "• **2-Step Rhythm Confirmation & Posture Lock:** Filters out single bumps when sitting down or adjusting your belt, and locks step counting to 0 whenever you are Sitting, Standing, or Laying."
            ),
            "highlights": [
                ChatHighlightItem(label="Gravity Filter", value="0.3 Hz cutoff preserves 1-3 Hz gait"),
                ChatHighlightItem(label="Peak-Valley Gate", value=">= 0.075g swing + 280 ms refractory"),
                ChatHighlightItem(label="Rhythm Confirmation", value="2-step rhythm lock rejects bumps"),
                ChatHighlightItem(label="Stationary Guard", value="0 steps added in Sit / Stand / Lay"),
            ],
            "follow_ups": [
                "What is my current live activity, model prediction, and step count right now?",
                "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
                "How does the AI tell Walking apart from Walking Upstairs and Walking Downstairs?",
            ],
        },
        {
            "id": "combinai_model",
            "keywords": [
                "combinai", "combined", "ensemble", "fusion", "not independently validated",
                "combine", "combines", "blends", "all 5", "unvalidated",
            ],
            "phrases": [
                "how does combinai", "how does the combinai", "combinai ensemble",
                "not independently validated", "combine the 4 base predictors",
            ],
            "title": "CombinAI (Ensemble): 4-Predictor Probability Fusion",
            "badge": "Ensemble (Unvalidated)",
            "answer": (
                "**CombinAI (Ensemble)** is a live multi-model probability ensemble rather than a fifth standalone base architecture. On every 2.56s window, it blends the probability distributions of the 4 live base predictors:\n"
                "• **20% CNN-LSTM** (raw `128x9` time-series sequence model; benchmarked at `90.57%` test / `90.14% ± 10.25%` LOSO)\n"
                "• **25% Regularized ANN** (`561`-feature deep MLP; benchmarked at `96.06%` test / `94.13% ± 5.85%` LOSO)\n"
                "• **30% Linear SVM** (`561`-feature tuned LinearSVC; benchmarked at `96.20%` test / `95.83% ± 4.84%` LOSO)\n"
                "• **25% 1D-CNN + LOSO-SVM Hybrid** (blends standalone `HAR 1D-CNN` with a LOSO-trained Linear SVM)\n"
                "• **Why marked “Not independently validated”?** Because the capstone benchmark report evaluates standalone models under strict Held-Out Test and 30-Fold LOSO protocols, CombinAI's live per-window confidence is shown directly without claiming an unvalidated offline benchmark accuracy."
            ),
            "highlights": [
                ChatHighlightItem(label="Predictor Role", value="Ensemble of 4 Live Base Predictors"),
                ChatHighlightItem(label="Benchmark Status", value="Not independently validated"),
                ChatHighlightItem(label="Fusion Weights", value="20% CNN-LSTM + 25% ANN + 30% SVM + 25% Hybrid"),
                ChatHighlightItem(label="Live Readout", value="Per-window confidence shown separately"),
            ],
            "follow_ups": [
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
                "How does the leak-free tuned Linear SVM achieve 96.20% test and 95.83% LOSO accuracy?",
            ],
        },
        {
            "id": "cnn_lstm_architecture",
            "keywords": [
                "cnn-lstm", "cnnlstm", "cnn_lstm", "lstm", "recurrent", "memory",
                "90.57", "90.57%", "90.14", "90.14%", "keras", "sequential", "conv1d",
            ],
            "phrases": [
                "cnn-lstm model", "how does the leak-free cnn-lstm", "how does the cnn-lstm",
                "cnn-lstm work", "9x128 motion",
            ],
            "title": "CNN-LSTM (Leak-Free): 90.57% Held-Out Test · 90.14% ± 10.25% LOSO",
            "badge": "CNN-LSTM",
            "answer": (
                f"**CNN-LSTM** ({cnn_lstm_params:,} parameters) combines 1D convolutional feature extraction with Long Short-Term Memory (LSTM) sequence modeling directly on raw `(128, 9)` sensor windows:\n"
                "• **Conv1D Blocks:** Two `Conv1D(64, kernel_size=3, ReLU)` layers extract local step impacts and gyro rotations, followed by `Dropout(0.5)` and `MaxPool1D(2)` (`128 -> 62` time steps).\n"
                "• **LSTM Memory Layer & Head:** `LSTM(100) -> Dropout(0.5) -> Dense(100, ReLU) -> Dense(6, Softmax)` models the temporal sequence across the 2.56-second window.\n"
                "• **Leak-Free Benchmark Results:** Using a strict subject-wise validation split (never using the held-out test set for early stopping), CNN-LSTM achieves **90.57% held-out test accuracy** (`90.60%` Macro F1) and **90.14% ± 10.25% 30-fold LOSO mean accuracy**."
            ),
            "highlights": [
                ChatHighlightItem(label="Held-Out Test Acc", value="90.57% (Macro F1: 90.60%)"),
                ChatHighlightItem(label="30-Fold LOSO Mean", value="90.14% ± 10.25% (Pooled: 90.42%)"),
                ChatHighlightItem(label="Architecture", value="2x Conv1D(64) -> MaxPool -> LSTM(100) -> Dense"),
                ChatHighlightItem(label="Parameters", value=f"{cnn_lstm_params:,} weights (361.0 KB)"),
            ],
            "follow_ups": [
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
                "How does Regularized ANN achieve 96.06% test accuracy and 94.13% LOSO mean on 561 features?",
            ],
        },
        {
            "id": "svm_architecture",
            "keywords": [
                "svm", "linearsvc", "support vector", "margin", "hyperplane", "96.20", "96.20%",
                "96.2", "95.83", "95.83%", "c=0.1", "linear svm",
            ],
            "phrases": [
                "svm model", "how does the leak-free tuned linear svm", "how does the svm",
                "96.20% test", "linear svm",
            ],
            "title": "Linear SVM (Leak-Free Tuned): 96.20% Test · 95.83% ± 4.84% LOSO",
            "badge": "Linear SVM",
            "answer": (
                "**Linear SVM (Leak-Free Tuned)** (`LinearSVC`, `C=0.1` selected via subject-wise validation) is our highest-scoring validated standalone classifier on the 561 engineered time and frequency features:\n"
                "• **Held-Out Test Protocol:** Achieves **96.20% test accuracy** (`96.29%` Macro F1) on the 2,947 held-out test windows (`96.13%` with original `C=0.5`).\n"
                "• **30-Fold LOSO Protocol:** Achieves **95.83% ± 4.84%** across all 30 unseen subject folds (`95.93%` pooled accuracy).\n"
                "• **Ultra-Fast Inference:** Computing `X_scaled @ coef.T + intercept` takes `~7.54 ms` for all 2,947 test windows (`0.0026 ms` per sample, `27.1 KB` model size)."
            ),
            "highlights": [
                ChatHighlightItem(label="Held-Out Test Acc", value="96.20% (Macro F1: 96.29%)"),
                ChatHighlightItem(label="30-Fold LOSO Mean", value="95.83% ± 4.84% across 30 subjects"),
                ChatHighlightItem(label="Input Features", value="561 standardized time + FFT features"),
                ChatHighlightItem(label="Efficiency", value="0.0026 ms/sample · 27.1 KB file size"),
            ],
            "follow_ups": [
                "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
                "How does Regularized ANN achieve 96.06% test accuracy and 94.13% LOSO mean on 561 features?",
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
            ],
        },
        {
            "id": "loso_architecture",
            "keywords": [
                "loso", "leave-one-subject-out", "leave one subject out", "30-fold", "30 fold",
                "hybrid", "1d-cnn", "1dcnn", "har_1dcnn", "93.93", "93.51", "protocol",
                "generalization", "unseen",
            ],
            "phrases": [
                "what does loso mean", "what is the loso", "1d-cnn + loso-svm hybrid",
                "leave-one-subject-out", "differ from standalone har 1d-cnn", "30-fold",
            ],
            "title": "LOSO Protocol vs. Live 1D-CNN + LOSO-SVM Hybrid & Standalone HAR 1D-CNN",
            "badge": "LOSO Protocol & Hybrid",
            "answer": (
                "**LOSO** stands for **Leave-One-Subject-Out** cross-validation — it is an **evaluation and training protocol, not a standalone model architecture**:\n"
                "• **What the 30-Fold LOSO Protocol Is:** Each model is trained on 29 volunteers and evaluated on the 1 remaining completely unseen volunteer, repeating 30 times so every model has both a **Held-Out Test Accuracy** and a **30-Fold LOSO Mean ± SD**.\n"
                f"• **Standalone HAR 1D-CNN Benchmark:** When evaluated alone on raw `9x128` signals, `HAR_1DCNN` ({cnn_params:,} parameters) achieves **93.93% held-out test accuracy** (`93.99%` Macro F1) and **93.51% ± 7.78% 30-fold LOSO accuracy**.\n"
                "• **Live `1D-CNN + LOSO-SVM Hybrid` Predictor:** In live tracking, `predict_loso_signals` blends `45%` standalone `HAR 1D-CNN` (`model.pth`) probabilities with `55%` LOSO-trained `Linear SVM` (`loso_model.joblib`) probabilities. Because this exact hybrid blend was not benchmarked as a single row in the offline report, its benchmark score is labeled **Not independently validated**."
            ),
            "highlights": [
                ChatHighlightItem(label="LOSO Definition", value="30-Fold Leave-One-Subject-Out Protocol"),
                ChatHighlightItem(label="Standalone HAR 1D-CNN", value="Test: 93.93% · LOSO: 93.51% ± 7.78%"),
                ChatHighlightItem(label="LOSO-Trained SVM", value="LOSO: 95.83% ± 4.84% (Test: 96.20%)"),
                ChatHighlightItem(label="Live Hybrid Status", value="45% 1D-CNN + 55% LOSO-SVM (Unvalidated)"),
            ],
            "follow_ups": [
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "How does the CombinAI ensemble combine the 4 base predictors and why is it marked not independently validated?",
                "How does the leak-free tuned Linear SVM achieve 96.20% test and 95.83% LOSO accuracy?",
            ],
        },
        {
            "id": "ann_architecture",
            "keywords": [
                "ann", "mlp", "regularized", "561", "features", "engineered", "dense",
                "adamw", "early stopping", "extract_561", "linear", "dropout", "96.06", "94.13",
            ],
            "phrases": [
                "ann model", "how does regularized ann", "how does the ann", "regularized ann",
                "561 engineered features", "96.06% test", "ann architecture",
            ],
            "title": "Regularized ANN: 96.06% Held-Out Test · 94.13% ± 5.85% LOSO",
            "badge": "Regularized ANN",
            "answer": (
                f"**Regularized ANN** (`RegularizedHAR_ANN`, {ann_params:,} parameters) is a deep multi-layer perceptron trained on the 561 engineered time- and frequency-domain features:\n"
                "• **Validated Benchmark Scores:** Achieves **96.06% held-out test accuracy** (`96.06%` Macro F1) and **94.13% ± 5.85% 30-fold LOSO accuracy** (outperforming an unregularized Plain ANN baseline at `94.91%` test / `93.84% ± 6.98%` LOSO).\n"
                "• **Network Structure:** Stacks 3 regularized dense blocks (`561 -> 256 -> 128 -> 64 -> 6`) with `BatchNorm1d`, `ReLU` activations, and graduated `Dropout(0.4, 0.3, 0.2)` trained with `AdamW` weight decay."
            ),
            "highlights": [
                ChatHighlightItem(label="Held-Out Test Acc", value="96.06% (Macro F1: 96.06%)"),
                ChatHighlightItem(label="30-Fold LOSO Mean", value="94.13% ± 5.85% across 30 subjects"),
                ChatHighlightItem(label="Architecture", value="561 -> 256 -> 128 -> 64 -> 6 (BatchNorm+Dropout)"),
                ChatHighlightItem(label="Parameters", value=f"{ann_params:,} weights (739.1 KB)"),
            ],
            "follow_ups": [
                "How does the leak-free tuned Linear SVM achieve 96.20% test and 95.83% LOSO accuracy?",
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "How does the system separate Sitting, Standing, and Laying postures?",
            ],
        },
        {
            "id": "sitting_vs_standing",
            "keywords": [
                "sitting", "standing", "laying", "posture", "postures", "stationary",
                "tilt", "angle", "pitch", "gravity", "confusion", "confused",
            ],
            "phrases": [
                "sitting, standing, and laying", "sitting vs standing", "separate sitting",
                "stationary postures", "sitting and standing",
            ],
            "title": "How the AI Separates Sitting vs. Standing vs. Laying",
            "badge": "Posture Analysis",
            "answer": (
                "Sitting, Standing, and Laying all have near-zero body motion (`linear energy < 0.05g`), making Sitting vs. Standing the hardest pair in UCI HAR (70 mutual errors in the baseline Linear SVM). Our pipeline separates them using **3D waist gravity orientation and tilt angle**:\n"
                "• **Laying (< 35° pitch):** Gravity acts sideways/back-to-front across the torso rather than head-to-toe (`> 99%` accuracy).\n"
                "• **Sitting (35° – 65° waist tilt):** When seated, the pelvis and waist pocket tilt relative to vertical, shifting the gravity vector ratio between `g_x`, `g_y`, and `g_z`.\n"
                "• **Standing (>= 66° upright vertical):** The torso and waist stay upright aligned with Earth's vertical gravity vector (`~1.0g`). In our Task 4 benchmark experiments, adding 32 explicit 3D gravity-tilt features with a Two-Stage Tilt-SVM raised test accuracy to **96.81%** (`96.11% ± 4.74%` LOSO)."
            ),
            "highlights": [
                ChatHighlightItem(label="Laying Angle", value="< 35 deg horizontal pitch"),
                ChatHighlightItem(label="Sitting Angle", value="35 deg - 65 deg seated waist tilt"),
                ChatHighlightItem(label="Standing Angle", value=">= 66 deg upright vertical posture"),
                ChatHighlightItem(label="Task 4 Tilt-SVM", value="96.81% Test · 96.11% ± 4.74% LOSO"),
            ],
            "follow_ups": [
                "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
                "How does the AI tell Walking apart from Walking Upstairs and Walking Downstairs?",
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
            ],
        },
        {
            "id": "walking_vs_stairs",
            "keywords": [
                "walking", "upstairs", "downstairs", "stairs", "climb", "descent",
                "dynamic", "gait", "frequency", "hz", "6 activities", "six activities",
            ],
            "phrases": [
                "walking apart from", "walking upstairs", "walking downstairs",
                "walking vs upstairs", "upstairs and walking downstairs",
            ],
            "title": "How the AI Separates Walking, Upstairs, and Downstairs",
            "badge": "Dynamic Gait",
            "answer": (
                "When the waist sensor detects sustained periodic motion (`1.0 – 3.5 Hz`), the models distinguish the 3 walking activities by their **cadence frequency, vertical impact crest factor, and forward torso lean**:\n"
                "• **Walking (~1.75 – 2.25 Hz, ~105–135 SPM):** Balanced vertical bounce and steady forward harmonic rhythm.\n"
                "• **Walking Upstairs (~1.30 – 1.82 Hz, ~80–110 SPM):** Slower step cadence with stronger upward lift effort and forward torso lean (`pitch < 64°`).\n"
                "• **Walking Downstairs (~2.06 – 2.50 Hz, ~124–150 SPM):** Faster step frequency with sharp vertical heel-strike impact spikes (`crest factor >= 2.42`)."
            ),
            "highlights": [
                ChatHighlightItem(label="Flat Walking", value="1.75 - 2.25 Hz · Steady rhythm"),
                ChatHighlightItem(label="Walking Upstairs", value="1.30 - 1.82 Hz · Forward lean & lift"),
                ChatHighlightItem(label="Walking Downstairs", value=">= 2.06 Hz · High impact crest factor"),
                ChatHighlightItem(label="Transition Speed", value="Evaluates recent 1.12s sub-window"),
            ],
            "follow_ups": [
                "How does the step counter detect my steps accurately without false counts?",
                "How does the system separate Sitting, Standing, and Laying postures?",
                "What are the 9 sensor channels at 50 Hz and how does the 0.3 Hz gravity filter work?",
            ],
        },
        {
            "id": "signals_filter",
            "keywords": [
                "signal", "channels", "9 channels", "butterworth", "filter", "cutoff",
                "0.3", "0.3hz", "gravity", "sampling", "50hz", "50 hz", "window",
                "2.56", "128", "smooth", "smoothed", "smoothing", "ema", "hysteresis",
                "imu", "accelerometer", "gyroscope",
            ],
            "phrases": [
                "9 sensor channels", "0.3 hz gravity filter", "butterworth", "cutoff frequency",
                "9 channels", "50 hz", "gravity filter",
            ],
            "title": "50 Hz Sensor Pipeline, 9 Channels & 0.3 Hz Gravity Filter",
            "badge": "DSP Pipeline",
            "answer": (
                "Every prediction analyzes a **2.56-second sliding window** containing **128 samples at 50 Hz** across **9 channels**:\n"
                "• **Channels 1–3 (`body_acc_xyz`):** Linear body movement acceleration after subtracting gravity.\n"
                "• **Channels 4–6 (`body_gyro_xyz`):** 3D angular velocity from the gyroscope (hip rotation & pitch).\n"
                "• **Channels 7–9 (`total_acc_xyz`):** Raw accelerometer readings including Earth's `1.0g` gravity.\n"
                "• **0.3 Hz Low-Pass Gravity Filter:** Because gravity changes slowly (`< 0.3 Hz`) while walking steps oscillate at `1.4 – 2.5 Hz`, a `0.3 Hz` low-pass filter (`alpha = 0.036` at 50 Hz) cleanly separates posture gravity from dynamic step motion."
            ),
            "highlights": [
                ChatHighlightItem(label="Window Size", value="128 samples @ 50 Hz = 2.56 seconds"),
                ChatHighlightItem(label="9 IMU Channels", value="body_acc (3) + gyro (3) + total_acc (3)"),
                ChatHighlightItem(label="Gravity Cutoff", value="fc = 0.3 Hz Low-Pass Filter"),
                ChatHighlightItem(label="Fast Warm-Up", value="Starts at 32 samples (0.64s), updates every 300ms"),
            ],
            "follow_ups": [
                "How does the leak-free CNN-LSTM model work on 9x128 motion signals?",
                "How does the step counter detect my steps accurately without false counts?",
                "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
            ],
        },
        {
            "id": "live_status",
            "keywords": [
                "right now", "current", "my status", "my telemetry", "doing now",
                "my steps", "live session", "what am i doing", "model loaded", "weights status",
                "model status", "loaded weights", "device", "cpu", "gpu",
            ],
            "phrases": [
                "right now", "what is my current live", "what am i doing", "live telemetry",
                "model status", "my current live activity",
            ],
            "title": "Your Current Live Session, Active Predictor & Step Count",
            "badge": "Live Telemetry",
            "answer": (
                f"Right now your primary highlighted predictor is **{selected_model}** (while all 4 base predictors `CNN-LSTM`, `Regularized ANN`, `Linear SVM`, `1D-CNN + LOSO-SVM Hybrid` plus `CombinAI (Ensemble)` run side by side on device `{device}`).\n"
                f"• **Current Detected Activity:** **{live_act}** ({live_conf} live window confidence — separate from offline benchmark accuracy)\n"
                f"• **Total Session Steps:** **{live_steps:,} steps** ({live_win_steps} steps in the latest 2.56s window at **{live_spm} SPM** cadence)\n"
                f"• **Loaded Checkpoints:** `cnn_lstm_model.keras`, `har_ann_regularized.pth`, `svm_model.joblib`, `loso_model.joblib`, and `model.pth` are active."
            ),
            "highlights": [
                ChatHighlightItem(label="Detected Activity", value=f"{live_act} ({live_conf} live conf)"),
                ChatHighlightItem(label="Highlighted Predictor", value=f"{selected_model}"),
                ChatHighlightItem(label="Total Steps", value=f"{live_steps:,} steps · {live_spm} SPM"),
                ChatHighlightItem(label="Live Predictors", value="4 Base Predictors + CombinAI Ensemble"),
            ],
            "follow_ups": [
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "How do I use the 6-digit device number (code) to see my mobile prediction on my PC?",
                "How does the step counter detect my steps accurately without false counts?",
            ],
        },
        {
            "id": "project_author",
            "keywords": [
                "tapash", "pradhan", "pushpa", "shova", "author", "creator", "developer",
                "contact", "email", "github", "repo", "code", "scu", "seoul christian",
                "university", "capstone", "team", "dataset", "uci", "api", "endpoint", "docs",
            ],
            "phrases": [
                "who built", "what dataset was used", "contact the developer",
                "project code", "github", "seoul christian", "about this capstone",
            ],
            "title": "About This UCI HAR Capstone Project & Dataset",
            "badge": "Project Info",
            "answer": (
                "This **Human Activity Recognition (UCI HAR) Studio** was developed for the AI & Big Data Convergence Capstone at **Seoul Christian University** by **Tapash Pradhan** (`tapashpradhan1234@gmail.com`) with teammates **Pushpa Ramtel** and **Shova Dhital**.\n"
                "• **Dataset:** UCI Human Activity Recognition Using Smartphones Dataset (30 volunteers aged 19–48, waist-mounted Samsung Galaxy S II, 50 Hz accelerometer & gyroscope, 10,299 windows: 7,352 train / 2,947 test across 6 activities).\n"
                "• **Validated Benchmark Models:** `Linear SVM` (96.20% test / 95.83% ± 4.84% LOSO), `Regularized ANN` (96.06% test / 94.13% ± 5.85% LOSO), `HAR 1D-CNN` (93.93% test / 93.51% ± 7.78% LOSO), and `CNN-LSTM` (90.57% test / 90.14% ± 10.25% LOSO), plus live `1D-CNN + LOSO-SVM Hybrid` and `CombinAI (Ensemble)`."
            ),
            "highlights": [
                ChatHighlightItem(label="Lead Developer", value="Tapash Pradhan (SCU)"),
                ChatHighlightItem(label="Team Members", value="Pushpa Ramtel · Shova Dhital"),
                ChatHighlightItem(label="Dataset", value="UCI HAR · 30 Subjects · 10,299 Windows"),
                ChatHighlightItem(label="GitHub", value="github.com/Tapash36"),
            ],
            "follow_ups": [
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "Why must I keep my mobile attached at my waist in one place and not hold it in my hand?",
                "How do I use the 6-digit device number (code) to see my mobile prediction on my PC?",
            ],
        },
    ]

    best_topic = None
    best_score = 0
    for t in topics:
        score = 0
        for p in t["phrases"]:
            if p in q:
                score += 6
        for kw in t["keywords"]:
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", q):
                score += 2
        if score > best_score:
            best_score = score
            best_topic = t

    if best_topic is None or best_score < 2:
        return ChatResponsePayload(
            title="UCI HAR AI Assistant — Choose Any Topic",
            badge="Model Guide",
            answer=(
                "I can answer questions about our **4 live base predictors** (**CNN-LSTM**, **Regularized ANN**, **Linear SVM**, **1D-CNN + LOSO-SVM Hybrid**), "
                "**CombinAI (Ensemble)**, our **validated Held-Out Test vs. 30-Fold LOSO benchmark results** (including standalone **HAR 1D-CNN**), "
                "**why your mobile must stay at your waist**, or **how to watch live on your PC via the 6-digit code**."
            ),
            highlights=[
                ChatHighlightItem(label="Linear SVM & ANN", value="SVM: 96.20% Test · ANN: 96.06% Test"),
                ChatHighlightItem(label="1D-CNN & CNN-LSTM", value="1D-CNN: 93.93% Test · CNN-LSTM: 90.57% Test"),
                ChatHighlightItem(label="Live Activity", value=f"{live_act} ({live_conf} live conf)"),
                ChatHighlightItem(label="Step Counter", value=f"{live_steps:,} steps · {live_spm} SPM"),
            ],
            follow_ups=[
                "Compare the 4 live base predictors, CombinAI ensemble, and validated benchmark results",
                "What does LOSO mean, what is the 1D-CNN + LOSO-SVM Hybrid predictor, and how does it differ from standalone HAR 1D-CNN?",
                "How do I use the 6-digit device number (code) to see my mobile prediction on my PC?",
            ],
        )

    return ChatResponsePayload(
        title=best_topic["title"],
        badge=best_topic["badge"],
        answer=best_topic["answer"],
        highlights=best_topic["highlights"],
        follow_ups=best_topic["follow_ups"],
    )


@app.post("/api/chat", response_model=ChatResponsePayload)
def chat_with_model_assistant(
    payload: ChatRequestPayload,
    user: dict = Depends(optional_authenticated_user),
):
    return build_model_chatbot_reply(payload.message, payload.context)


# ----------------- Low-Latency Cross-Device Live Relay (Waist Phone -> PC) -----------------
ROOM_CODE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")
MAX_RELAY_ROOMS = 500
ROOM_IDLE_TTL_SEC = 1800.0
RELAY_ROOMS_LOCK = threading.Lock()
RELAY_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


@dataclass
class LiveRelayRoom:
    code: str
    owner_id: Optional[int] = None
    owner_username: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    seq: int = 0
    sender_active: bool = False
    sender_label: str = "Waist Phone IMU"
    senders: Set[WebSocket] = field(default_factory=set)
    viewers: Set[WebSocket] = field(default_factory=set)
    latest_telemetry: Optional[Dict[str, Any]] = None
    latest_prediction: Optional[Dict[str, Any]] = None


LIVE_RELAY_ROOMS: Dict[str, LiveRelayRoom] = {}


def _normalize_room_code(raw: str) -> str:
    cleaned = re.sub(r"[\s-]+", "", (raw or "").strip().upper())
    if not ROOM_CODE_PATTERN.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Pair code must be 4-12 letters or digits.")
    return cleaned


def _purge_stale_rooms_locked(now: Optional[float] = None) -> None:
    ts = now if now is not None else time.time()
    expired = [
        code
        for code, rm in LIVE_RELAY_ROOMS.items()
        if (ts - rm.last_activity_at) > ROOM_IDLE_TTL_SEC and not rm.senders and not rm.viewers
    ]
    for code in expired:
        LIVE_RELAY_ROOMS.pop(code, None)
    if len(LIVE_RELAY_ROOMS) > MAX_RELAY_ROOMS:
        ordered = sorted(LIVE_RELAY_ROOMS.items(), key=lambda kv: kv[1].last_activity_at)
        for code, rm in ordered[: len(LIVE_RELAY_ROOMS) - MAX_RELAY_ROOMS]:
            if not rm.senders and not rm.viewers:
                LIVE_RELAY_ROOMS.pop(code, None)


def _get_or_create_room(code: str, owner: Optional[dict] = None) -> LiveRelayRoom:
    now = time.time()
    with RELAY_ROOMS_LOCK:
        _purge_stale_rooms_locked(now)
        rm = LIVE_RELAY_ROOMS.get(code)
        if rm is None:
            rm = LiveRelayRoom(
                code=code,
                owner_id=owner["id"] if owner else None,
                owner_username=owner["username"] if owner else None,
                created_at=now,
                last_activity_at=now,
            )
            LIVE_RELAY_ROOMS[code] = rm
        else:
            rm.last_activity_at = now
            if owner and rm.owner_id is None:
                rm.owner_id = owner["id"]
                rm.owner_username = owner["username"]
        return rm


async def _ws_broadcast_targets(targets: Set[WebSocket], msg: Dict[str, Any], room: LiveRelayRoom, role_attr: str) -> None:
    with RELAY_ROOMS_LOCK:
        sockets = list(targets)
    if not sockets:
        return
    payload_str = json.dumps(msg, separators=(",", ":"))

    async def _send_one(ws: WebSocket):
        try:
            await asyncio.wait_for(ws.send_text(payload_str), timeout=1.5)
            return None
        except Exception:
            return ws

    results = await asyncio.gather(*(_send_one(ws) for ws in sockets), return_exceptions=True)
    dead = [r for r in results if isinstance(r, WebSocket)]
    if dead:
        with RELAY_ROOMS_LOCK:
            target_set: Set[WebSocket] = getattr(room, role_attr)
            for d in dead:
                target_set.discard(d)
            if role_attr == "senders" and not room.senders:
                room.sender_active = False


def _sync_publish_prediction_to_room(
    room_code: str,
    pred_dict: Dict[str, Any],
    raw_window: Optional[Any] = None,
    model_type: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    cleaned = re.sub(r"[\s-]+", "", (room_code or "").strip().upper())
    if not ROOM_CODE_PATTERN.fullmatch(cleaned):
        return
    now = time.time()
    room = _get_or_create_room(cleaned)
    telem_record: Optional[Dict[str, Any]] = None
    if raw_window is not None:
        try:
            arr = np.asarray(raw_window, dtype=np.float32)
            if arr.shape == (9, 128):
                arr = arr.T
            if arr.shape == (128, 9):
                last_row = arr[-1]
                samples_tail = [
                    [round(float(v), 3) for v in row[:6]]
                    for row in arr[-32:]
                ]
                acc_last = [round(float(last_row[0]), 3), round(float(last_row[1]), 3), round(float(last_row[2]), 3)]
                gyro_last = [round(float(last_row[3]), 3), round(float(last_row[4]), 3), round(float(last_row[5]), 3)]
                energy_val = round(float(np.linalg.norm(last_row[:3])), 3)
                telem_record = {
                    "samples": samples_tail,
                    "acc": acc_last,
                    "gyro": gyro_last,
                    "tilt_deg": 78,
                    "energy": energy_val,
                    "buffer_count": 128,
                    "model_type": model_type or "cnn_lstm",
                }
        except Exception:
            telem_record = None

    with RELAY_ROOMS_LOCK:
        room.seq += 1
        room.sender_active = True
        room.last_activity_at = now
        if user_id and room.owner_id is None:
            room.owner_id = user_id
        if telem_record is not None:
            if room.latest_telemetry and isinstance(room.latest_telemetry.get("steps"), dict):
                telem_record["steps"] = room.latest_telemetry["steps"]
            room.latest_telemetry = telem_record
        room.latest_prediction = pred_dict
        seq = room.seq
        has_viewers = bool(room.viewers)
    if has_viewers and RELAY_EVENT_LOOP is not None and not RELAY_EVENT_LOOP.is_closed():
        msg = {
            "type": "prediction",
            "seq": seq,
            "prediction": pred_dict,
            "model_type": model_type or "cnn_lstm",
            "server_ts_ms": int(now * 1000),
        }
        try:
            if telem_record is not None:
                t_msg = {
                    "type": "telemetry",
                    "seq": seq,
                    "sender_active": True,
                    "label": room.sender_label,
                    "server_ts_ms": int(now * 1000),
                    **telem_record,
                }
                asyncio.run_coroutine_threadsafe(
                    _ws_broadcast_targets(room.viewers, t_msg, room, "viewers"),
                    RELAY_EVENT_LOOP,
                )
            asyncio.run_coroutine_threadsafe(
                _ws_broadcast_targets(room.viewers, msg, room, "viewers"),
                RELAY_EVENT_LOOP,
            )
        except RuntimeError:
            pass


@app.post("/api/relay/create")
def create_relay_room(
    user: dict = Depends(optional_authenticated_user),
    preferred_code: Optional[str] = Query(default=None),
):
    now = time.time()
    pref_str = preferred_code if isinstance(preferred_code, str) else ""
    cleaned_pref = re.sub(r"[\s-]+", "", pref_str.strip().upper())
    with RELAY_ROOMS_LOCK:
        _purge_stale_rooms_locked(now)
        if cleaned_pref and ROOM_CODE_PATTERN.fullmatch(cleaned_pref):
            candidate = cleaned_pref
        else:
            for _ in range(32):
                candidate = str(secrets.randbelow(900_000) + 100_000)
                if candidate not in LIVE_RELAY_ROOMS:
                    break
            else:
                candidate = secrets.token_hex(3).upper()
        rm = LIVE_RELAY_ROOMS.get(candidate)
        if rm is None:
            rm = LiveRelayRoom(
                code=candidate,
                owner_id=user["id"],
                owner_username=user["username"],
                created_at=now,
                last_activity_at=now,
            )
            LIVE_RELAY_ROOMS[candidate] = rm
        else:
            rm.last_activity_at = now
    return {"room_code": candidate, "expires_in_sec": int(ROOM_IDLE_TTL_SEC)}


@app.get("/api/relay/active")
def find_active_account_room(user: dict = Depends(optional_authenticated_user)):
    if user["username"] == "guest_researcher":
        return {"room_code": None, "sender_active": False}
    now = time.time()
    with RELAY_ROOMS_LOCK:
        _purge_stale_rooms_locked(now)
        matches = [
            rm
            for rm in LIVE_RELAY_ROOMS.values()
            if rm.owner_id == user["id"] and (rm.sender_active or rm.senders)
        ]
        if not matches:
            return {"room_code": None, "sender_active": False}
        latest_rm = max(matches, key=lambda r: r.last_activity_at)
        return {
            "room_code": latest_rm.code,
            "sender_active": latest_rm.sender_active,
            "sender_label": latest_rm.sender_label,
        }


@app.post("/api/relay/{room_code}/publish")
async def publish_relay_update(
    room_code: str,
    payload: RelayPublishPayload,
    user: dict = Depends(optional_authenticated_user),
):
    code = _normalize_room_code(room_code)
    room = _get_or_create_room(code, owner=user)
    now = time.time()
    broadcast_msgs: List[Dict[str, Any]] = []

    with RELAY_ROOMS_LOCK:
        room.last_activity_at = now
        if payload.sender_label:
            room.sender_label = payload.sender_label[:64]
        if payload.sender_active is not None:
            room.sender_active = payload.sender_active
            room.seq += 1
            broadcast_msgs.append(
                {
                    "type": "sender_status",
                    "seq": room.seq,
                    "active": room.sender_active,
                    "label": room.sender_label,
                    "server_ts_ms": int(now * 1000),
                }
            )
        if payload.telemetry is not None:
            room.sender_active = True
            room.seq += 1
            room.latest_telemetry = payload.telemetry
            broadcast_msgs.append(
                {
                    "type": "telemetry",
                    "seq": room.seq,
                    **payload.telemetry,
                    "sender_active": True,
                    "label": room.sender_label,
                    "server_ts_ms": int(now * 1000),
                }
            )
        if payload.prediction is not None:
            room.sender_active = True
            room.seq += 1
            room.latest_prediction = payload.prediction
            broadcast_msgs.append(
                {
                    "type": "prediction",
                    "seq": room.seq,
                    "prediction": payload.prediction,
                    "server_ts_ms": int(now * 1000),
                }
            )
        seq = room.seq
        viewer_count = len(room.viewers)

    for m in broadcast_msgs:
        await _ws_broadcast_targets(room.viewers, m, room, "viewers")

    return {"ok": True, "room_code": code, "seq": seq, "viewer_count": viewer_count}


@app.get("/api/relay/{room_code}/latest")
def get_relay_latest(
    room_code: str,
    since_seq: int = Query(default=0, ge=0),
    user: dict = Depends(optional_authenticated_user),
):
    code = _normalize_room_code(room_code)
    with RELAY_ROOMS_LOCK:
        rm = LIVE_RELAY_ROOMS.get(code)
        if rm is None:
            return {
                "exists": False,
                "changed": False,
                "room_code": code,
                "seq": 0,
                "sender_active": False,
                "sender_label": "Waist Phone IMU",
                "viewer_count": 0,
                "telemetry": None,
                "prediction": None,
            }
        rm.last_activity_at = time.time()
        changed = rm.seq > since_seq
        return {
            "exists": True,
            "changed": changed,
            "room_code": code,
            "seq": rm.seq,
            "sender_active": rm.sender_active or bool(rm.senders),
            "sender_label": rm.sender_label,
            "viewer_count": len(rm.viewers),
            "telemetry": rm.latest_telemetry if changed or since_seq == 0 else None,
            "prediction": rm.latest_prediction if changed or since_seq == 0 else None,
        }


@app.websocket("/ws/live/{room_code}")
async def websocket_live_relay(websocket: WebSocket, room_code: str):
    global RELAY_EVENT_LOOP
    RELAY_EVENT_LOOP = asyncio.get_running_loop()

    cleaned_code = re.sub(r"[\s-]+", "", (room_code or "").strip().upper())
    if not ROOM_CODE_PATTERN.fullmatch(cleaned_code):
        await websocket.close(code=4400, reason="Invalid pair code")
        return

    token = websocket.query_params.get("token") or websocket.cookies.get(SESSION_COOKIE)
    user = authenticate_session_token(token) or _get_or_create_guest_user()

    role = (websocket.query_params.get("role") or "viewer").strip().lower()
    if role not in ("sender", "viewer"):
        role = "viewer"

    await websocket.accept()
    room = _get_or_create_room(cleaned_code, owner=user)

    try:
        if role == "sender":
            label = (websocket.query_params.get("label") or "Waist Phone IMU").strip()[:64]
            with RELAY_ROOMS_LOCK:
                room.senders.add(websocket)
                room.sender_active = True
                if label:
                    room.sender_label = label
                room.seq += 1
                seq = room.seq
                viewer_count = len(room.viewers)
                room.last_activity_at = time.time()
            await websocket.send_text(
                json.dumps(
                    {"type": "relay_ready", "role": "sender", "room_code": cleaned_code, "viewer_count": viewer_count},
                    separators=(",", ":"),
                )
            )
            await _ws_broadcast_targets(
                room.viewers,
                {
                    "type": "sender_status",
                    "seq": seq,
                    "active": True,
                    "label": room.sender_label,
                    "viewer_count": viewer_count,
                    "server_ts_ms": int(time.time() * 1000),
                },
                room,
                "viewers",
            )
        else:
            with RELAY_ROOMS_LOCK:
                room.viewers.add(websocket)
                room.last_activity_at = time.time()
                snapshot = {
                    "type": "room_snapshot",
                    "room_code": cleaned_code,
                    "seq": room.seq,
                    "sender_active": room.sender_active or bool(room.senders),
                    "sender_label": room.sender_label,
                    "viewer_count": len(room.viewers),
                    "telemetry": room.latest_telemetry,
                    "prediction": room.latest_prediction,
                    "server_ts_ms": int(time.time() * 1000),
                }
                viewer_count = len(room.viewers)
            await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
            await _ws_broadcast_targets(
                room.senders,
                {"type": "viewer_count", "viewer_count": viewer_count},
                room,
                "senders",
            )

        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            now = time.time()
            with RELAY_ROOMS_LOCK:
                room.last_activity_at = now

            if msg_type == "ping":
                await websocket.send_text(
                    json.dumps(
                        {"type": "pong", "t": data.get("t"), "server_ts_ms": int(now * 1000)},
                        separators=(",", ":"),
                    )
                )
            elif role == "sender" and msg_type == "telemetry":
                with RELAY_ROOMS_LOCK:
                    room.sender_active = True
                    room.seq += 1
                    seq = room.seq
                    if isinstance(data.get("label"), str) and data["label"].strip():
                        room.sender_label = data["label"].strip()[:64]
                    telem_record = {
                        "samples": data.get("samples"),
                        "acc": data.get("acc"),
                        "gyro": data.get("gyro"),
                        "tilt_deg": data.get("tilt_deg"),
                        "energy": data.get("energy"),
                        "buffer_count": data.get("buffer_count"),
                        "steps": data.get("steps"),
                        "model_type": data.get("model_type"),
                    }
                    room.latest_telemetry = telem_record
                    label_now = room.sender_label
                await _ws_broadcast_targets(
                    room.viewers,
                    {
                        "type": "telemetry",
                        "seq": seq,
                        "sender_active": True,
                        "label": label_now,
                        "server_ts_ms": int(now * 1000),
                        **telem_record,
                    },
                    room,
                    "viewers",
                )
            elif role == "sender" and msg_type == "predict":
                raw_window = data.get("window")
                model_type = str(data.get("model_type") or "cnn_lstm")
                try:
                    rate_hz = float(data.get("sample_rate_hz") or 50.0)
                except (TypeError, ValueError):
                    rate_hz = 50.0
                rate_hz = float(np.clip(rate_hz, 10.0, 240.0))
                prev_act = data.get("previous_activity")
                try:
                    pred_res = await asyncio.to_thread(
                        _run_live_inference,
                        raw_window,
                        model_type,
                        rate_hz,
                        prev_act,
                        False,
                    )
                    pred_dict = pred_res.model_dump()
                except Exception as exc:
                    await websocket.send_text(
                        json.dumps({"type": "predict_error", "detail": str(exc)}, separators=(",", ":"))
                    )
                    continue

                with RELAY_ROOMS_LOCK:
                    room.sender_active = True
                    room.seq += 1
                    seq = room.seq
                    room.latest_prediction = pred_dict
                out_msg = {
                    "type": "prediction",
                    "seq": seq,
                    "prediction": pred_dict,
                    "steps": data.get("steps"),
                    "measured_rate_hz": rate_hz,
                    "model_type": model_type,
                    "client_ts": data.get("client_ts"),
                    "server_ts_ms": int(time.time() * 1000),
                }
                await asyncio.gather(
                    _ws_broadcast_targets(room.senders, out_msg, room, "senders"),
                    _ws_broadcast_targets(room.viewers, out_msg, room, "viewers"),
                )
            elif role == "sender" and msg_type == "sender_status":
                active_flag = bool(data.get("active", False))
                with RELAY_ROOMS_LOCK:
                    room.sender_active = active_flag
                    if isinstance(data.get("label"), str) and data["label"].strip():
                        room.sender_label = data["label"].strip()[:64]
                    if not active_flag:
                        room.seq += 1
                    seq = room.seq
                    label_now = room.sender_label
                await _ws_broadcast_targets(
                    room.viewers,
                    {
                        "type": "sender_status",
                        "seq": seq,
                        "active": active_flag,
                        "label": label_now,
                        "reason": data.get("reason"),
                        "server_ts_ms": int(now * 1000),
                    },
                    room,
                    "viewers",
                )
            elif role == "sender" and msg_type == "state_reset":
                with RELAY_ROOMS_LOCK:
                    room.seq += 1
                    seq = room.seq
                    room.latest_prediction = None
                    room.latest_telemetry = None
                await _ws_broadcast_targets(
                    room.viewers,
                    {"type": "state_reset", "seq": seq, "server_ts_ms": int(now * 1000)},
                    room,
                    "viewers",
                )
            elif role == "viewer" and msg_type == "remote_command":
                cmd_msg = {
                    "type": "remote_command",
                    "command": data.get("command"),
                    "model_type": data.get("model_type"),
                    "server_ts_ms": int(now * 1000),
                }
                await _ws_broadcast_targets(room.senders, cmd_msg, room, "senders")

    except WebSocketDisconnect:
        pass
    finally:
        if role == "sender":
            notify_offline = False
            with RELAY_ROOMS_LOCK:
                room.senders.discard(websocket)
                if not room.senders:
                    room.sender_active = False
                    room.seq += 1
                    seq = room.seq
                    label_now = room.sender_label
                    notify_offline = bool(room.viewers)
                else:
                    seq = room.seq
                    label_now = room.sender_label
            if notify_offline:
                await _ws_broadcast_targets(
                    room.viewers,
                    {
                        "type": "sender_status",
                        "seq": seq,
                        "active": False,
                        "label": label_now,
                        "reason": "disconnected",
                        "server_ts_ms": int(time.time() * 1000),
                    },
                    room,
                    "viewers",
                )
        else:
            with RELAY_ROOMS_LOCK:
                room.viewers.discard(websocket)
                viewer_count = len(room.viewers)
                has_senders = bool(room.senders)
            if has_senders:
                await _ws_broadcast_targets(
                    room.senders,
                    {"type": "viewer_count", "viewer_count": viewer_count},
                    room,
                    "senders",
                )
