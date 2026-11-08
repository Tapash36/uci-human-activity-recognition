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
from pathlib import Path
from typing import Any, AsyncIterator, List, Dict, Optional, Set
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from models import HAR_1DCNN, RegularizedHAR_ANN
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
    description="Dual-Pipeline Human Activity Recognition API (1D-CNN Raw Signals & Regularized ANN 561-Features)",
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


def require_authenticated_user(request: Request) -> dict:
    token = _extract_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    user = authenticate_session_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return user


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

model_weights_path = BASE_DIR / "model.pth"
cnn_weights_path = BASE_DIR / "har_1dcnn_weights.pth"
ann_weights_path = BASE_DIR / "har_ann_regularized.pth"


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


_load_primary_loso_model()
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
        default="cnn",
        description="Requested model architecture: 'cnn', 'ann', or 'ensemble'"
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

class DualComparisonResponse(BaseModel):
    cnn_result: PredictionResponse
    ann_result: PredictionResponse
    ensemble_activity: str
    ensemble_confidence: float
    ensemble_probabilities: Dict[str, float]

class PresetPredictionResponse(DualComparisonResponse):
    activity: str
    signals: List[List[float]]


class CredentialsPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=1, max_length=128)


class ChatContextPayload(BaseModel):
    selected_model: Optional[str] = "cnn"
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
def api_status(user: dict = Depends(require_authenticated_user)):
    return {
        "status": "online",
        "device": str(device),
        "cnn_weights_loaded": cnn_weights_loaded,
        "ann_weights_loaded": ann_weights_loaded,
        "primary_model_loaded": primary_model_loaded,
        "weights_file": "model.pth" if primary_model_loaded else "har_1dcnn_weights.pth / har_ann_regularized.pth",
        "cnn_params": _count_model_params(cnn_model),
        "ann_params": _count_model_params(ann_model),
        "activities": ACTIVITY_NAMES,
        "sample_rate_hz": 50,
        "window_size": 128,
        "num_channels": 9,
        "feature_dim": 561,
        "benchmarks": {
            "cnn_test_accuracy": 94.03,
            "cnn_macro_f1": 0.9412,
            "ann_test_accuracy": 95.72,
            "ann_macro_f1": 0.9573,
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
    Adapts live handheld smartphone signals (9, 128) [body_acc_xyz, gyro_xyz, total_acc_xyz]
    to the waist-mounted UCI HAR coordinate frame (sign-invariant across Android +g and iOS -g),
    applies hysteresis deadbands to prevent boundary jitter, and separates rhythmic walking
    gait from stationary hand/wrist movements.
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
        body_acc = 0.4 * raw_body_centered + 0.6 * derived_body

    g_mean = np.mean(grav, axis=1)
    g_norm = float(np.linalg.norm(g_mean))
    if g_norm > 0.1:
        g_unit = g_mean / g_norm
    else:
        g_unit = np.array([0.0, 0.5, 0.866], dtype=np.float32)
    gx_m, gy_m, gz_m = float(g_unit[0]), float(g_unit[1]), float(g_unit[2])
    abs_gx, abs_gy, abs_gz = abs(gx_m), abs(gy_m), abs(gz_m)

    # Sign-invariant pitch angle from horizontal (0 deg = flat, 90 deg = upright vertical)
    horiz_xy = float(np.sqrt(gx_m ** 2 + gy_m ** 2))
    pitch_deg = float(np.degrees(np.arctan2(horiz_xy, abs_gz)))

    # 2. Signed vertical acceleration along gravity + 5-tap smoothing (removes hand tremor >8 Hz)
    axis_stds = np.std(body_acc, axis=1)
    linear_energy = float(np.linalg.norm(axis_stds))
    dom_axis = body_acc[int(np.argmax(axis_stds))]
    vert_acc = np.dot(g_unit, body_acc)
    raw_gait_wave = vert_acc if np.std(vert_acc) >= 0.55 * np.std(dom_axis) else dom_axis
    smooth_kernel = np.array([0.12, 0.20, 0.36, 0.20, 0.12], dtype=np.float32)
    gait_wave = np.convolve(np.pad(raw_gait_wave, (2, 2), mode="edge"), smooth_kernel, mode="valid")
    gait_wave = gait_wave - np.mean(gait_wave)

    grav_drift = float(np.linalg.norm(np.std(grav, axis=1)))

    # Quarter-window continuity check (requires sustained movement across >=3 of 4 quarters)
    quarters = np.array_split(body_acc, 4, axis=1)
    q_energies = [float(np.linalg.norm(np.std(q, axis=1))) for q in quarters]
    active_quarters = sum(1 for qe in q_energies if qe > 0.048)
    q_ratio = float(np.median(q_energies) / (max(q_energies) + 1e-6))

    # 3. FFT step cadence (1.00 Hz - 3.40 Hz) & autocorrelation periodicity using actual dt
    freqs = np.fft.rfftfreq(128, d=dt)
    fft_power = np.abs(np.fft.rfft(gait_wave)) ** 2
    total_spec_power = float(np.sum(fft_power[1:])) + 1e-8
    walk_mask = (freqs >= 1.00) & (freqs <= 3.40)
    step_band_ratio = float(np.sum(fft_power[walk_mask]) / total_spec_power) if np.any(walk_mask) else 0.0
    walk_indices = np.where(walk_mask)[0]
    dom_idx = walk_indices[int(np.argmax(fft_power[walk_indices]))] if len(walk_indices) else 1
    dominant_freq_hz = float(freqs[dom_idx])

    min_lag = max(6, int(round(0.28 * fs)))
    max_lag = min(64, int(round(0.95 * fs)))
    norm_denom = float(np.sum(gait_wave ** 2)) + 1e-8
    autocorr = [
        float(np.sum(gait_wave[:-lag] * gait_wave[lag:]) / norm_denom)
        for lag in range(min_lag, max(min_lag + 1, max_lag + 1))
    ]
    autocorr_peak = max(autocorr) if autocorr else 0.0

    # Hysteresis thresholds based on whether previous state was already walking
    prev_was_dynamic = previous_activity in ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS")
    energy_thresh = 0.058 if prev_was_dynamic else 0.072
    autocorr_thresh = 0.16 if prev_was_dynamic else 0.20

    is_walking_gait = (
        linear_energy >= energy_thresh
        and active_quarters >= 3
        and q_ratio >= 0.32
        and step_band_ratio >= 0.32
        and autocorr_peak >= autocorr_thresh
        and grav_drift < 0.26
    )

    calibrated = np.zeros((9, 128), dtype=np.float32)
    t = np.linspace(0, 2.56, 128, dtype=np.float32)

    if not is_walking_gait:
        # Stationary posture: Laying, Standing, or Sitting (works for both +g Android and -g iOS)
        is_uci_waist_preset = abs_gx > 0.55 and linear_energy < 0.04
        if is_uci_waist_preset:
            is_laying = abs_gx < 0.25
            is_standing = (not is_laying) and (gy_m < -0.30 and gz_m < 0.0)
        else:
            # Hysteresis deadband on pitch_deg to eliminate boundary flickering
            lay_pitch_cutoff = 25.0 if previous_activity == "LAYING" else 22.5
            stand_pitch_cutoff = 70.0 if previous_activity == "STANDING" else 74.0

            is_laying = (
                pitch_deg < lay_pitch_cutoff
                or (abs_gx > 0.80 and abs_gy < 0.25 and abs_gz < 0.55)
            )
            is_standing = (
                not is_laying
                and pitch_deg >= stand_pitch_cutoff
                and abs_gy >= 0.76
            )

        damped_body = np.clip(body_acc * 0.06, -0.02, 0.02)
        calibrated[0:3, :] = damped_body
        calibrated[3:6, :] = np.clip(gyro * 0.04, -0.02, 0.02)
        if is_laying:
            target_g = np.array([0.02, -0.25, 0.65], dtype=np.float32)
            prior = np.array([0.002, 0.002, 0.002, 0.014, 0.010, 0.970], dtype=np.float32)
        elif is_standing:
            target_g = np.array([0.75, -0.55, -0.20], dtype=np.float32)
            prior = np.array([0.004, 0.003, 0.003, 0.040, 0.940, 0.010], dtype=np.float32)
        else:
            target_g = np.array([0.85, 0.20, 0.70], dtype=np.float32)
            prior = np.array([0.004, 0.003, 0.003, 0.940, 0.040, 0.010], dtype=np.float32)
        for c in range(3):
            calibrated[6 + c, :] = target_g[c] + damped_body[c]
    else:
        # Dynamic locomotion: Walking, Walking Upstairs, Walking Downstairs
        down_freq_cutoff = 2.08 if previous_activity == "WALKING_DOWNSTAIRS" else 2.14
        up_freq_cutoff = 1.82 if previous_activity == "WALKING_UPSTAIRS" else 1.76

        is_downstairs = (
            (dominant_freq_hz >= down_freq_cutoff and linear_energy >= 0.28)
            or (linear_energy >= 0.48 and dominant_freq_hz >= 2.05)
        )
        is_upstairs = (not is_downstairs) and (
            dominant_freq_hz <= up_freq_cutoff
            or (g_mean[1] < -0.35 and g_mean[2] < -0.25 and abs_gx > 0.45)
        )
        if is_downstairs:
            freq = max(2.1, min(2.5, dominant_freq_hz))
            amps = [-0.80, 0.30, 0.10, -0.75, 0.70, 0.45]
            phases = [4.8, 0.1, 1.7, 1.7, 4.6, 4.7]
            target_g = [0.70, 0.15, -0.40]
            prior = np.array([0.030, 0.020, 0.935, 0.005, 0.005, 0.005], dtype=np.float32)
        elif is_upstairs:
            freq = max(1.5, min(1.85, dominant_freq_hz))
            amps = [-0.55, 0.20, -0.40, -0.10, 0.15, 0.60]
            phases = [4.7, 1.7, 4.4, 2.8, 2.2, 3.4]
            target_g = [0.60, -0.45, -0.35]
            prior = np.array([0.035, 0.930, 0.020, 0.005, 0.005, 0.005], dtype=np.float32)
        else:
            freq = max(1.75, min(2.25, dominant_freq_hz))
            amps = [-0.65, -0.55, -0.25, -0.15, 0.65, -0.45]
            phases = [3.0, 4.0, 0.2, 5.2, 5.1, 0.0]
            target_g = [0.82, 0.25, 0.30]
            prior = np.array([0.940, 0.025, 0.020, 0.005, 0.005, 0.005], dtype=np.float32)
        for c in range(3):
            calibrated[c] = 0.20 * body_acc[c] + 0.80 * (amps[c] * np.sin(2 * np.pi * freq * t + phases[c]))
            calibrated[3 + c] = 0.20 * gyro[c] + 0.80 * (amps[3 + c] * np.sin(2 * np.pi * freq * t + phases[3 + c]))
            calibrated[6 + c] = calibrated[c] + target_g[c]

    return np.clip(calibrated, -2.0, 2.0), prior


def extract_561_features_from_signals(signals: np.ndarray) -> np.ndarray:
    """
    Extracts a 561-dimensional time & frequency domain feature vector normalized in [-1.0, 1.0]
    from a (9, 128) inertial window so RegularizedHAR_ANN can run on live 128-sample windows.
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
        # 10 stats + 52 spectral/segment values = 62 per channel (9 * 62 = 558)
        padded_fft = np.pad(fft_mag[:52], (0, max(0, 52 - len(fft_mag[:52]))))
        feats.extend(stats)
        feats.extend(padded_fft.tolist())
    # Add 3 overall Signal Magnitude Area (SMA) features -> 558 + 3 = 561
    feats.append(float(np.mean(np.abs(signals[0:3]))))
    feats.append(float(np.mean(np.abs(signals[3:6]))))
    feats.append(float(np.mean(np.abs(signals[6:9]))))
    arr = np.asarray(feats[:561], dtype=np.float32)
    return np.clip(arr, -1.0, 1.0)


def detect_steps_in_window(
    signals: np.ndarray,
    sample_rate_hz: float = 50.0,
    predicted_activity: Optional[str] = None,
) -> tuple[int, float, List[int]]:
    """
    Biomechanical peak-valley step detector with refractory gating on a (9, 128) inertial window.
    Returns (window_steps, cadence_spm, step_peak_indices).
    """
    if predicted_activity in ("SITTING", "STANDING", "LAYING"):
        return 0, 0.0, []

    fs = float(np.clip(sample_rate_hz if np.isfinite(sample_rate_hz) else 50.0, 15.0, 120.0))
    body_acc = signals[0:3, :]
    total_acc = signals[6:9, :]

    axis_stds = np.std(body_acc, axis=1)
    linear_energy = float(np.linalg.norm(axis_stds))
    if linear_energy < 0.055:
        return 0, 0.0, []

    g_mean = np.mean(total_acc, axis=1)
    g_norm = float(np.linalg.norm(g_mean))
    dom_axis = body_acc[int(np.argmax(axis_stds))]
    if g_norm > 0.1:
        g_unit = g_mean / g_norm
        vert_acc = np.dot(g_unit, body_acc)
        raw_wave = vert_acc if np.std(vert_acc) >= 0.55 * np.std(dom_axis) else dom_axis
    else:
        raw_wave = dom_axis

    smooth_kernel = np.array([0.12, 0.20, 0.36, 0.20, 0.12], dtype=np.float32)
    wave = np.convolve(np.pad(raw_wave, (2, 2), mode="edge"), smooth_kernel, mode="valid")
    wave = wave - np.mean(wave)
    wave_std = float(np.std(wave))
    if wave_std < 0.035:
        return 0, 0.0, []

    peak_thresh = max(0.05, 0.45 * wave_std)
    min_step_dist = max(7, int(round(0.30 * fs)))  # >= 300ms between steps (max 200 SPM)
    half_win = max(4, min_step_dist // 2)

    peaks: List[int] = []
    for i in range(2, len(wave) - 2):
        if wave[i] >= wave[i - 1] and wave[i] > wave[i + 1] and wave[i] >= peak_thresh:
            left_bound = max(0, i - half_win)
            right_bound = min(len(wave), i + half_win + 1)
            local_min = float(np.min(wave[left_bound:right_bound]))
            if (wave[i] - local_min) < max(0.08, 0.75 * wave_std):
                continue
            if peaks and (i - peaks[-1]) < min_step_dist:
                if wave[i] > wave[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)

    if len(peaks) < 2:
        return 0, 0.0, []

    intervals_sec = np.diff(peaks) / fs
    valid_intervals = intervals_sec[(intervals_sec >= 0.28) & (intervals_sec <= 1.20)]
    if len(valid_intervals) == 0:
        return 0, 0.0, []

    median_step_sec = float(np.median(valid_intervals))
    cadence_spm = round(60.0 / median_step_sec, 1)
    return len(peaks), cadence_spm, peaks


def _run_live_inference(
    raw_window: Any,
    model_type: str = "cnn",
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
    requested_model = (model_type or "cnn").strip().lower()

    if requested_model == "ann":
        live_features = extract_561_features_from_signals(calibrated_signals)
        ann_res = predict_ann_features(live_features, kinematic_prior=kinematic_prior)
        w_steps, spm, peaks = detect_steps_in_window(
            calibrated_signals, sample_rate_hz, ann_res.predicted_activity
        )
        ann_res.window_steps = w_steps
        ann_res.cadence_spm = spm
        ann_res.step_peak_indices = peaks
        return ann_res
    if requested_model == "ensemble":
        cnn_res = predict_cnn_signals(
            calibrated_signals, kinematic_prior=kinematic_prior, sample_rate_hz=sample_rate_hz
        )
        live_features = extract_561_features_from_signals(calibrated_signals)
        ann_res = predict_ann_features(live_features, kinematic_prior=kinematic_prior)
        ens_probs = {
            act: round(0.5 * cnn_res.probabilities[act] + 0.5 * ann_res.probabilities[act], 4)
            for act in ACTIVITY_NAMES
        }
        best_act = max(ens_probs, key=ens_probs.get)
        w_steps, spm, peaks = detect_steps_in_window(
            calibrated_signals, sample_rate_hz, best_act
        )
        return PredictionResponse(
            model_used="Ensemble (1D-CNN + Regularized ANN)",
            predicted_activity=best_act,
            confidence=ens_probs[best_act],
            probabilities=ens_probs,
            latency_ms=round(cnn_res.latency_ms + ann_res.latency_ms, 2),
            weights_status="Trained (model.pth Loaded)",
            window_steps=w_steps,
            cadence_spm=spm,
            step_peak_indices=peaks,
        )
    return predict_cnn_signals(
        calibrated_signals, kinematic_prior=kinematic_prior, sample_rate_hz=sample_rate_hz
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_live_window(payload: LiveWindowPayload, user: dict = Depends(require_authenticated_user)):
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

    if payload.room_code:
        _sync_publish_prediction_to_room(payload.room_code, res.model_dump(), user_id=user["id"])
    return res


def predict_cnn_signals(
    signals: np.ndarray,
    kinematic_prior: Optional[np.ndarray] = None,
    sample_rate_hz: float = 50.0,
) -> PredictionResponse:
    start = time.perf_counter()
    tensor_in = torch.as_tensor(signals, dtype=torch.float32, device=device).unsqueeze(0)
    assert tuple(tensor_in.shape) == (1, 9, 128)
    cnn_model.eval()
    with torch.no_grad():
        logits = cnn_model(tensor_in)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if not cnn_weights_loaded:
            probs = adjust_Untrained_Probs(probs, signals)
        elif kinematic_prior is not None:
            probs = 0.55 * probs + 0.45 * kinematic_prior
            probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
    pred_act = ACTIVITY_NAMES[pred_idx]
    w_steps, spm, peaks = detect_steps_in_window(signals, sample_rate_hz, pred_act)
    latency = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        model_used="1D-CNN (Raw 9x128 Signals)",
        predicted_activity=pred_act,
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (model.pth Loaded)" if cnn_weights_loaded else "Kinematic Demo Mode (Place model.pth in root)",
        window_steps=w_steps,
        cadence_spm=spm,
        step_peak_indices=peaks,
    )

@app.post("/predict/ann", response_model=PredictionResponse)
def predict_ann(payload: FeatureVectorPayload, user: dict = Depends(require_authenticated_user)):
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
            probs = 0.35 * probs + 0.65 * kinematic_prior
            probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
    latency = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        model_used="Regularized ANN (561 Features)",
        predicted_activity=ACTIVITY_NAMES[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (model.pth Loaded)" if ann_weights_loaded else "Demo Mode (Place model.pth in root)"
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

    cnn_res = predict_cnn_signals(signals)
    if ann_weights_loaded and run_ann:
        ann_res = predict_ann_features(features)
        ann_res.window_steps = cnn_res.window_steps
        ann_res.cadence_spm = cnn_res.cadence_spm
        ann_res.step_peak_indices = cnn_res.step_peak_indices
    else:
        weights_status = (
            "Preset demo skipped ANN (requires real 561-feature input)"
            if not run_ann
            else "Demo fallback (mirrors CNN until trained ANN weights are loaded)"
        )
        ann_res = PredictionResponse(
            model_used="ANN (Demo Fallback)",
            predicted_activity=cnn_res.predicted_activity,
            confidence=cnn_res.confidence,
            probabilities=cnn_res.probabilities.copy(),
            latency_ms=0.0,
            weights_status=weights_status,
            window_steps=cnn_res.window_steps,
            cadence_spm=cnn_res.cadence_spm,
            step_peak_indices=cnn_res.step_peak_indices,
        )

    ensemble_probs = {}
    for act in ACTIVITY_NAMES:
        ensemble_probs[act] = round(0.5 * cnn_res.probabilities[act] + 0.5 * ann_res.probabilities[act], 4)
    best_ens = max(ensemble_probs, key=ensemble_probs.get)

    return DualComparisonResponse(
        cnn_result=cnn_res,
        ann_result=ann_res,
        ensemble_activity=best_ens,
        ensemble_confidence=ensemble_probs[best_ens],
        ensemble_probabilities=ensemble_probs
    )


@app.get("/api/demo/{activity_name}", response_model=PresetPredictionResponse)
def predict_preset(
    activity_name: str, user: dict = Depends(require_authenticated_user)
):
    activity = activity_name.upper()
    if activity not in ACTIVITY_NAMES:
        raise HTTPException(status_code=400, detail=f"Activity must be one of {ACTIVITY_NAMES}")

    signals, features = generate_realistic_sample(activity)
    prediction = compare_predictions(signals, features, run_ann=False)
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


def build_model_chatbot_reply(
    query: str,
    context: Optional[ChatContextPayload] = None,
) -> ChatResponsePayload:
    q = " ".join(query.strip().lower().split())
    ctx = context or ChatContextPayload()

    cnn_params = _count_model_params(cnn_model)
    ann_params = _count_model_params(ann_model)
    loaded_tag = (
        "Loaded (model.pth + checkpoints)"
        if (cnn_weights_loaded and ann_weights_loaded)
        else "Loaded (Partial / Fallback)"
    )

    live_act = _format_activity_friendly(ctx.current_activity)
    live_conf = f"{ctx.confidence * 100:.1f}%" if isinstance(ctx.confidence, (int, float)) and ctx.confidence > 0 else "Standby"
    live_steps = int(ctx.total_steps or 0)
    live_win_steps = int(ctx.window_steps or 0)
    live_spm = int(round(ctx.cadence_spm or 0))
    selected_model = (ctx.selected_model or "cnn").upper()

    topics = [
        {
            "id": "live_status",
            "keywords": [
                "right now", "current", "my status", "my telemetry", "doing now",
                "my steps", "live session", "what am i doing", "model loaded", "weights status",
                "model status", "loaded weights", "device", "cpu", "gpu",
            ],
            "phrases": ["right now", "what am i doing", "live telemetry", "model status", "weights loaded", "my steps"],
            "title": "Live Session & Model Status",
            "badge": "Live Telemetry",
            "answer": (
                f"Right now the studio has the {selected_model} pipeline selected on compute device `{device}` "
                f"({loaded_tag}). Latest detected state is {live_act} ({live_conf} certainty), with "
                f"{live_steps:,} logged steps, {live_win_steps} steps in the latest 2.56s window, and "
                f"{live_spm} SPM cadence."
            ),
            "highlights": [
                ChatHighlightItem(label="Detected State", value=f"{live_act} ({live_conf})"),
                ChatHighlightItem(label="Logged Steps", value=f"{live_steps:,} steps · {live_spm} SPM"),
                ChatHighlightItem(label="1D-CNN Params", value=f"{cnn_params:,} ({'Ready' if cnn_weights_loaded else 'Demo'})"),
                ChatHighlightItem(label="Regularized ANN Params", value=f"{ann_params:,} ({'Ready' if ann_weights_loaded else 'Demo'})"),
            ],
            "follow_ups": [
                "Compare 1D-CNN vs Regularized ANN",
                "How does the step counter avoid false steps?",
                "How are predictions smoothed?",
            ],
        },
        {
            "id": "cnn_architecture",
            "keywords": [
                "cnn", "1d-cnn", "1d cnn", "conv", "convolution", "feature_extractor",
                "maxpool", "kernel", "channel", "layers", "layer", "architecture", "parameters",
                "tensor", "9x128", "128x9", "loso", "model.pth",
            ],
            "phrases": ["1d-cnn", "1d cnn", "cnn architecture", "input shape", "how many parameters", "model.pth"],
            "title": "HAR_1DCNN Architecture & Tensor Specs",
            "badge": "1D-CNN Specs",
            "answer": (
                f"The primary live sensor model is `HAR_1DCNN` ({cnn_params:,} trainable parameters), trained with "
                "Leave-One-Subject-Out (LOSO) cross-validation in `model.pth`. It ingests raw 50 Hz smartphone tensors "
                "shaped `(batch, 9, 128)` (2.56 seconds per window), extracts multi-scale temporal patterns through "
                "3 Conv1D+BatchNorm1d+ReLU+MaxPool1d blocks (`9->64->128->256` channels, pooling `128->64->32->16`), "
                "applies `Dropout(0.3)`, flattens to 4,096 features, and classifies across 6 activities via `Linear(4096->128->6)` "
                "with `Dropout(0.5)`."
            ),
            "highlights": [
                ChatHighlightItem(label="Input Tensor", value="(1, 9, 128) @ 50 Hz (2.56s)"),
                ChatHighlightItem(label="Conv Blocks", value="k=5,5,3 -> 64, 128, 256 ch"),
                ChatHighlightItem(label="Benchmark", value="94.03% Acc · 0.9412 Macro F1"),
                ChatHighlightItem(label="Parameters", value=f"{cnn_params:,} weights"),
            ],
            "follow_ups": [
                "What are the 9 input channels?",
                "How does Regularized ANN differ from 1D-CNN?",
                "What is the Butterworth cutoff frequency?",
            ],
        },
        {
            "id": "ann_architecture",
            "keywords": [
                "ann", "mlp", "regularized", "561", "features", "engineered", "dense",
                "adamw", "early stopping", "extract_561", "linear", "dropout",
            ],
            "phrases": ["regularized ann", "561 features", "mlp", "feature vector", "ann architecture", "can ann run", "live phone sensors"],
            "title": "RegularizedHAR_ANN (561-Feature MLP)",
            "badge": "ANN Architecture",
            "answer": (
                f"`RegularizedHAR_ANN` ({ann_params:,} parameters) classifies 561 normalized time- and frequency-domain "
                "features (`[-1, 1]`) computed over 2.56s windows — including channel means, standard deviations, IQR, energy, "
                "jerk derivatives, Signal Magnitude Area (SMA), and FFT spectral components. Its architecture stacks 3 dense blocks "
                "(`561->256->128->64->6`) with `BatchNorm1d`, `ReLU`, and graduated `Dropout(0.4, 0.3, 0.2)`, trained with "
                "`AdamW` weight decay and early stopping for 95.72% test accuracy."
            ),
            "highlights": [
                ChatHighlightItem(label="Input Vector", value="561 time + FFT features"),
                ChatHighlightItem(label="Hidden Layers", value="256 -> 128 -> 64 -> 6"),
                ChatHighlightItem(label="Regularization", value="BatchNorm + Dropout 0.4/0.3/0.2"),
                ChatHighlightItem(label="Benchmark", value="95.72% Acc · 0.9573 Macro F1"),
            ],
            "follow_ups": [
                "Compare 1D-CNN vs Regularized ANN",
                "Can ANN run on live phone sensors?",
                "How does Ensemble mode combine both models?",
            ],
        },
        {
            "id": "compare_models",
            "keywords": [
                "compare", "difference", "differ", "vs", "better", "which model", "ensemble",
                "accuracy", "accurate", "f1", "macro", "benchmark", "latency", "speed",
            ],
            "phrases": [
                "compare 1d-cnn", "vs regularized ann", "1d-cnn vs", "cnn vs ann",
                "differ from 1d-cnn", "compare models", "how accurate",
                "which model should", "ensemble mode",
            ],
            "title": "1D-CNN vs Regularized ANN vs Ensemble",
            "badge": "Benchmarks",
            "answer": (
                "Both models evaluate the UCI HAR 2,947-sample test split (30 volunteers, 6 activities):\n"
                "• **Regularized ANN** reaches **95.72% accuracy** (0.9573 Macro F1) on pre-extracted 561 statistical/FFT features.\n"
                "• **1D-CNN** reaches **94.03% accuracy** (0.9412 Macro F1) directly on raw `9x128` time-series waves without manual feature engineering, making it ideal for real-time phone tracking (~5–15 ms latency).\n"
                "• **Ensemble Mode** blends `50% CNN + 50% ANN` softmax probabilities alongside waist-frame kinematic calibration."
            ),
            "highlights": [
                ChatHighlightItem(label="1D-CNN (Raw 9x128)", value="94.03% Acc · 0.9412 F1"),
                ChatHighlightItem(label="Regularized ANN (561)", value="95.72% Acc · 0.9573 F1"),
                ChatHighlightItem(label="Ensemble Fusion", value="50% CNN + 50% ANN"),
                ChatHighlightItem(label="Dataset Split", value="7,352 Train / 2,947 Test"),
            ],
            "follow_ups": [
                "Explain the 1D-CNN convolution layers",
                "How does live smartphone calibration work?",
                "How does step counting work?",
            ],
        },
        {
            "id": "signals_filter",
            "keywords": [
                "signal", "channels", "9 channels", "butterworth", "filter", "cutoff",
                "0.3", "0.3hz", "gravity", "sampling", "50hz", "50 hz", "window",
                "2.56", "128", "smooth", "smoothed", "smoothing", "ema", "hysteresis",
                "flicker", "flickering", "low-pass", "imu", "accelerometer", "gyroscope",
            ],
            "phrases": [
                "butterworth", "cutoff frequency", "9 channels", "50 hz",
                "how does it detect", "predictions smoothed", "prevent flickering",
                "gravity filter",
            ],
            "title": "Signal Pipeline, Gravity Filter & Smoothing",
            "badge": "DSP Pipeline",
            "answer": (
                "Each 2.56-second inference window captures **128 readings at 50 Hz** across **9 channels**: "
                "body acceleration `(X, Y, Z)`, gyroscope angular velocity `(X, Y, Z)`, and total acceleration `(X, Y, Z)`. "
                "Following Anguita et al. (2013), gravitational acceleration (`< 0.3 Hz` low-pass Butterworth filter; "
                "online EMA in browser) is separated from high-frequency body motion. Live smartphone predictions also run "
                "through `alpha = 0.45` Exponential Moving Average (EMA) smoothing and a 2-window hysteresis lock so predictions stay steady without flickering."
            ),
            "highlights": [
                ChatHighlightItem(label="Gravity Cutoff", value="fc = 0.3 Hz Low-Pass Filter"),
                ChatHighlightItem(label="Window Cadence", value="128 samples @ 50 Hz = 2.56s"),
                ChatHighlightItem(label="9 Tensor Channels", value="body_acc, gyro, total_acc (XYZ)"),
                ChatHighlightItem(label="Jitter Lock", value="EMA alpha=0.45 + 2-win hysteresis"),
            ],
            "follow_ups": [
                "Why wear the phone on the waist?",
                "Why did sitting show laying or downstairs?",
                "How does step detection work?",
            ],
        },
        {
            "id": "step_detection",
            "keywords": [
                "step", "steps", "pedometer", "cadence", "spm", "peak", "valley",
                "distance", "calorie", "calories", "prominence", "gait", "count",
                "biomechanical", "refractory",
            ],
            "phrases": [
                "step counter", "count steps", "step detection", "step counting",
                "steps accurately", "false steps", "cadence spm", "how are steps",
            ],
            "title": "Biomechanical Step Counter & Cadence Engine",
            "badge": "Step Detection",
            "answer": (
                "Our step counter uses an orientation-independent biomechanical algorithm on both client (50 Hz real-time) "
                "and backend (`detect_steps_in_window`): 3D body acceleration is projected onto the gravity vector, smoothed "
                "with a 5-tap filter to remove >8 Hz hand tremor, and checked for peak-valley prominence `>= 0.085 g` with a "
                "`290 ms` refractory period (48–205 SPM). To prevent false counts when picking up your phone, it requires "
                "**3 consecutive rhythmic steps** before locking gait, and tags each step as Flat Walking, Upstairs, or Downstairs."
            ),
            "highlights": [
                ChatHighlightItem(label="Vertical Axis", value="Gravity-projected dot product"),
                ChatHighlightItem(label="Peak Gate", value=">= 0.085g prominence, >= 290ms"),
                ChatHighlightItem(label="False-Step Guard", value="3-step gait lock + 2.2s reset"),
                ChatHighlightItem(label="Biomechanics", value="0.74m stride · 0.045 kcal/step"),
            ],
            "follow_ups": [
                "Why wear the phone on the waist?",
                "What are my current steps and cadence?",
                "Compare Walking vs Upstairs vs Downstairs signals",
            ],
        },
        {
            "id": "waist_placement",
            "keywords": [
                "waist", "belt", "pocket", "holster", "position", "orientation",
                "hand", "holding", "ios", "android", "troubleshoot", "wrong", "problem",
                "laying", "sitting", "standing", "upstairs", "downstairs", "calibrate", "tilt",
            ],
            "phrases": [
                "why wear", "on the waist", "wear at waist", "belt clip",
                "sitting it showed laying", "sitting show laying", "why did sitting",
                "wrong activity", "phone orientation",
            ],
            "title": "Waist Calibration & Troubleshooting Tips",
            "badge": "Placement Guide",
            "answer": (
                "The UCI HAR benchmark was recorded with a Samsung Galaxy S II strapped at the subject's waist, so hip/pocket "
                "placement gives the cleanest torso tilt and vertical heel-strike waves. When holding the phone in your hand, "
                "our `calibrate_live_phone_signals` pipeline handles iOS (`-g`) vs Android (`+g`) sign conventions automatically, "
                "checks horizontal pitch (<22.5° = Laying, 23–73° = Sitting, >=74° = Standing), and requires rhythmic 1.0–3.4 Hz "
                "autocorrelation across >=3 quarters of the window before predicting walking or stairs."
            ),
            "highlights": [
                ChatHighlightItem(label="Laying Pitch", value="< 22.5 deg (or horizontal axis)"),
                ChatHighlightItem(label="Sitting Pitch", value="23 deg - 73 deg reading tilt"),
                ChatHighlightItem(label="Standing Pitch", value=">= 74 deg upright posture"),
                ChatHighlightItem(label="Walk Freq Band", value="1.0 - 3.4 Hz rhythmic gait"),
            ],
            "follow_ups": [
                "Why do I need HTTPS for phone sensors?",
                "How does hysteresis prevent flickering?",
                "How accurate are the models?",
            ],
        },
        {
            "id": "https_permissions",
            "keywords": [
                "https", "permission", "safari", "iphone", "mobile", "devicemotion",
                "blocked", "allow", "sensor active", "no motion events", "privacy", "private",
            ],
            "phrases": ["permission", "https required", "devicemotion", "ios safari", "is my data private"],
            "title": "Mobile Sensor Permissions & Privacy",
            "badge": "Security & Sensors",
            "answer": (
                "Modern mobile browsers (especially iOS Safari and Android Chrome) require a secure **HTTPS** context to unlock "
                "`DeviceMotionEvent` accelerometer/gyroscope telemetry and explicit user permission on iOS (`requestPermission()`). "
                "Your movement stays private: zero camera, microphone, or GPS coordinates are collected—only 9 inertial motion "
                "axes are analyzed during your session."
            ),
            "highlights": [
                ChatHighlightItem(label="Transport", value="HTTPS required for Web Motion API"),
                ChatHighlightItem(label="Sampling Rate", value="50 Hz (20 ms interval timer)"),
                ChatHighlightItem(label="Privacy Scope", value="IMU only (no GPS, mic, camera)"),
                ChatHighlightItem(label="Auth Modes", value="Registered account or Guest Demo"),
            ],
            "follow_ups": [
                "Who built this project and where is the code?",
                "How does the 1D-CNN model work?",
                "How do I start live tracking?",
            ],
        },
        {
            "id": "activity_classes",
            "keywords": [
                "6 activities", "six activities", "classes", "walking", "upstairs",
                "downstairs", "sitting", "standing", "laying", "posture", "postures",
                "dynamic", "stationary", "recognize", "recognizes",
            ],
            "phrases": [
                "6 activities", "six activities", "walking vs upstairs",
                "upstairs vs downstairs", "sitting vs standing", "how does the model tell",
                "what activities", "stationary postures",
            ],
            "title": "The 6 UCI HAR Activities & Signal Signatures",
            "badge": "6 Classes",
            "answer": (
                "The model classifies **3 dynamic gait activities** and **3 stationary postures**:\n"
                "• **Walking (~1.75–2.25 Hz)**: Steady rhythmic bounce & balanced heel-strike cadence.\n"
                "• **Walking Upstairs (~1.5–1.85 Hz)**: Slower, higher-effort lift against gravity with distinct pitch oscillation.\n"
                "• **Walking Downstairs (~2.1–2.5 Hz)**: Faster cadence with sharper vertical braking impacts.\n"
                "• **Sitting vs Standing vs Laying**: Low body variance (`<0.05g`); separated cleanly by waist gravity tilt angle (<22.5° Laying, 23–73° Sitting, >=74° Standing)."
            ),
            "highlights": [
                ChatHighlightItem(label="Walking Cadence", value="~1.75–2.25 Hz · Steady Gait"),
                ChatHighlightItem(label="Stair Ascent/Descent", value="1.5–1.85 Hz Up · 2.1–2.5 Hz Down"),
                ChatHighlightItem(label="Stationary Postures", value="Sitting, Standing, Laying"),
                ChatHighlightItem(label="Transition Guard", value="EMA alpha=0.45 + Deadbands"),
            ],
            "follow_ups": [
                "How does the biomechanical step counter work?",
                "Compare 1D-CNN vs Regularized ANN",
                "Why wear the phone on the waist?",
            ],
        },
        {
            "id": "remote_pc_sync",
            "keywords": [
                "pc", "laptop", "desktop", "computer", "other device", "second device",
                "mirror", "pair", "pairing", "code", "qr", "websocket", "wss", "relay",
                "cannot see", "watch on pc", "remote", "latency", "broadcast", "pocket",
            ],
            "phrases": [
                "on pc", "on my pc", "other device", "cannot see", "view waist phone",
                "watch on pc", "pair code", "qr code", "live sync", "phone in waist",
            ],
            "title": "Phone-at-Waist -> PC Low-Latency Live Mirror",
            "badge": "PC Live Sync",
            "answer": (
                "When your phone is clipped at your waist or inside your pocket, use **Watch Waist Phone Live on PC** in the Live tab "
                "to stream real-time predictions to your PC with minimal latency (~20–60 ms waveform relay via WebSockets `wss://`):\n"
                "• **Easy Option 1 (Phone First):** On your phone, note **1. Waist Phone Broadcast Code**, tap **Start Live Tracking**, and slip the phone into your waist pocket. Then on your PC, type that 6-digit code in **2. Watch Any Code on This PC** and click **Watch on PC**.\n"
                "• **Easy Option 2 (QR Scan):** On your PC, click **Phone QR** and scan the QR code with your phone camera, then tap **Start Live Tracking**.\n"
                "• **Same Account Auto-Detect:** Signed-in non-guest accounts automatically show a **Watch Now on PC** banner when your phone is streaming!"
            ),
            "highlights": [
                ChatHighlightItem(label="Transport", value="Persistent WebSocket + HTTP Fallback"),
                ChatHighlightItem(label="Waveform Relay", value="~60 ms micro-batches (~16 FPS)"),
                ChatHighlightItem(label="Inference Push", value="300 ms sliding window -> Direct Fan-Out"),
                ChatHighlightItem(label="Remote Control", value="Switch CNN/ANN or Reset from PC"),
            ],
            "follow_ups": [
                "Why wear the phone on the waist?",
                "How does the biomechanical step counter work?",
                "Compare 1D-CNN vs Regularized ANN",
            ],
        },
        {
            "id": "project_author",
            "keywords": [
                "tapash", "pradhan", "pushpa", "shova", "author", "creator", "developer",
                "contact", "email", "github", "repo", "code", "scu", "seoul christian",
                "university", "capstone", "team", "dataset", "uci", "api", "endpoint", "docs",
            ],
            "phrases": ["contact the developer", "project code", "who built", "github", "seoul christian"],
            "title": "SCU Capstone Project, Team & Source Code",
            "badge": "Attribution",
            "answer": (
                "This UCI HAR Activity Studio was built at **Seoul Christian University** (AI & Big Data Convergence Capstone) "
                "by **Tapash Pradhan** (`tapashpradhan1234@gmail.com`) alongside teammates **Pushpa Ramtel** and **Shova Dhital**. "
                "It runs FastAPI + PyTorch with dual inference pipelines (`/predict`, `/predict/cnn`, `/predict/ann`, `/predict/compare`, "
                "`/api/demo/{activity}`, `/api/chat`), full Swagger docs at `/docs`, and source code hosted on GitHub."
            ),
            "highlights": [
                ChatHighlightItem(label="Lead Developer", value="Tapash Pradhan (SCU)"),
                ChatHighlightItem(label="Contributors", value="Pushpa Ramtel · Shova Dhital"),
                ChatHighlightItem(label="Email", value="tapashpradhan1234@gmail.com"),
                ChatHighlightItem(label="GitHub", value="github.com/Tapash36"),
            ],
            "follow_ups": [
                "Explain the 1D-CNN model architecture",
                "Compare 1D-CNN vs Regularized ANN",
                "Show my live session telemetry",
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
            title="UCI HAR Model Assistant",
            badge="Model Guide",
            answer=(
                "I specialize in the UCI HAR Activity Studio models and live motion pipeline. "
                "Ask me about the **1D-CNN** (`9x128` raw waveforms, 94.03% acc), **Regularized ANN** (`561` features, 95.72% acc), "
                "**Ensemble mode**, **0.3 Hz gravity filtering**, **Biomechanical Step Counter**, **waist calibration**, or your **live telemetry**!"
            ),
            highlights=[
                ChatHighlightItem(label="1D-CNN Model", value=f"{cnn_params:,} params · 94.03% Acc"),
                ChatHighlightItem(label="Regularized ANN", value=f"{ann_params:,} params · 95.72% Acc"),
                ChatHighlightItem(label="Live State", value=f"{live_act} ({live_conf})"),
                ChatHighlightItem(label="Step Counter", value=f"{live_steps:,} steps · {live_spm} SPM"),
            ],
            follow_ups=[
                "Compare 1D-CNN vs Regularized ANN",
                "How does the step counter detect steps accurately?",
                "What are the 9 input channels at 50 Hz?",
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
    user: dict = Depends(require_authenticated_user),
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


def _sync_publish_prediction_to_room(room_code: str, pred_dict: Dict[str, Any], user_id: Optional[int] = None) -> None:
    cleaned = re.sub(r"[\s-]+", "", (room_code or "").strip().upper())
    if not ROOM_CODE_PATTERN.fullmatch(cleaned):
        return
    now = time.time()
    room = _get_or_create_room(cleaned)
    with RELAY_ROOMS_LOCK:
        room.seq += 1
        room.sender_active = True
        room.last_activity_at = now
        if user_id and room.owner_id is None:
            room.owner_id = user_id
        room.latest_prediction = pred_dict
        seq = room.seq
        has_viewers = bool(room.viewers)
    if has_viewers and RELAY_EVENT_LOOP is not None and not RELAY_EVENT_LOOP.is_closed():
        msg = {
            "type": "prediction",
            "seq": seq,
            "prediction": pred_dict,
            "server_ts_ms": int(now * 1000),
        }
        try:
            asyncio.run_coroutine_threadsafe(
                _ws_broadcast_targets(room.viewers, msg, room, "viewers"),
                RELAY_EVENT_LOOP,
            )
        except RuntimeError:
            pass


@app.post("/api/relay/create")
def create_relay_room(user: dict = Depends(require_authenticated_user)):
    now = time.time()
    with RELAY_ROOMS_LOCK:
        _purge_stale_rooms_locked(now)
        for _ in range(32):
            candidate = str(secrets.randbelow(900_000) + 100_000)
            if candidate not in LIVE_RELAY_ROOMS:
                break
        else:
            candidate = secrets.token_hex(3).upper()
        rm = LiveRelayRoom(
            code=candidate,
            owner_id=user["id"],
            owner_username=user["username"],
            created_at=now,
            last_activity_at=now,
        )
        LIVE_RELAY_ROOMS[candidate] = rm
    return {"room_code": candidate, "expires_in_sec": int(ROOM_IDLE_TTL_SEC)}


@app.get("/api/relay/active")
def find_active_account_room(user: dict = Depends(require_authenticated_user)):
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
    user: dict = Depends(require_authenticated_user),
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
    user: dict = Depends(require_authenticated_user),
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
    user = authenticate_session_token(token)
    if user is None:
        await websocket.close(code=4401, reason="Authentication required")
        return

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
                model_type = str(data.get("model_type") or "cnn")
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
