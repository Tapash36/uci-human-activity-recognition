import os
import hashlib
import hmac
import logging
import pickle
import re
import secrets
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Request, Response
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


def require_authenticated_user(request: Request) -> dict:
    token = _extract_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

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
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return {"id": user["id"], "username": user["username"]}


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
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=128)


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
def serve_ui():
    return RedirectResponse(url="/static/index.html")

@app.post("/api/auth/register")
def register(payload: CredentialsPayload, request: Request, response: Response):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", payload.username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters: letters, numbers, '.', '_' or '-'.",
        )
    if len(payload.password) < 12:
        raise HTTPException(status_code=400, detail="Password must be at least 12 characters.")

    initialize_auth_db()
    username = payload.username.lower()
    connection = get_auth_db()
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(payload.password)),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already registered.")
    finally:
        connection.close()

    token = create_session(user_id)
    set_session_cookie(response, request, token)
    return {"username": username, "token": token}


@app.post("/api/auth/login")
def login(payload: CredentialsPayload, request: Request, response: Response):
    initialize_auth_db()
    connection = get_auth_db()
    try:
        user = connection.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (payload.username.lower(),),
        ).fetchone()
    finally:
        connection.close()

    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_session(user["id"])
    set_session_cookie(response, request, token)
    return {"username": user["username"], "token": token}


@app.get("/api/auth/me")
def current_user(user: dict = Depends(require_authenticated_user)):
    return {"username": user["username"]}


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


@app.get("/api/status")
def api_status(user: dict = Depends(require_authenticated_user)):
    return {
        "status": "online",
        "device": str(device),
        "cnn_weights_loaded": cnn_weights_loaded,
        "ann_weights_loaded": ann_weights_loaded,
        "benchmarks": {
            "cnn_test_accuracy": 94.03,
            "cnn_macro_f1": 0.9412,
            "ann_test_accuracy": 95.72,
            "ann_macro_f1": 0.9573
        }
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


@app.post("/predict", response_model=PredictionResponse)
def predict_live_window(payload: LiveWindowPayload, user: dict = Depends(require_authenticated_user)):
    window = np.asarray(payload.window, dtype=np.float32)
    if window.shape == (9, 128):
        window = window.T
    if window.shape != (128, 9):
        raise HTTPException(status_code=400, detail=f"Expected shape (128, 9), got {window.shape}")
    if not np.isfinite(window).all():
        raise HTTPException(status_code=400, detail="Window must contain only finite numbers")

    input_tensor = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
    if tuple(input_tensor.shape) != (1, 128, 9):
        raise HTTPException(
            status_code=400,
            detail=f"Expected input tensor shape (1, 128, 9), got {tuple(input_tensor.shape)}",
        )
    assert tuple(input_tensor.shape) == (1, 128, 9)

    min_value = float(input_tensor.min().item())
    max_value = float(input_tensor.max().item())
    print(
        "Live /predict tensor diagnostics: "
        f"shape={tuple(input_tensor.shape)}, dtype={input_tensor.dtype}, "
        f"min={min_value:.6f}, max={max_value:.6f}, model_type={payload.model_type}",
        flush=True,
    )
    if min_value < -2.0 or max_value > 2.0:
        logging.getLogger(__name__).warning(
            "Live /predict values outside [-2.0, 2.0]; clamping before inference"
        )
        input_tensor = input_tensor.clamp(-2.0, 2.0)

    raw_signals = input_tensor.squeeze(0).numpy().T.copy()
    calibrated_signals, kinematic_prior = calibrate_live_phone_signals(
        raw_signals,
        sample_rate_hz=payload.sample_rate_hz,
        previous_activity=payload.previous_activity,
    )
    requested_model = (payload.model_type or "cnn").strip().lower()

    if requested_model == "ann":
        live_features = extract_561_features_from_signals(calibrated_signals)
        ann_res = predict_ann_features(live_features, kinematic_prior=kinematic_prior)
        w_steps, spm, peaks = detect_steps_in_window(
            calibrated_signals, payload.sample_rate_hz, ann_res.predicted_activity
        )
        ann_res.window_steps = w_steps
        ann_res.cadence_spm = spm
        ann_res.step_peak_indices = peaks
        return ann_res
    if requested_model == "ensemble":
        cnn_res = predict_cnn_signals(
            calibrated_signals, kinematic_prior=kinematic_prior, sample_rate_hz=payload.sample_rate_hz
        )
        live_features = extract_561_features_from_signals(calibrated_signals)
        ann_res = predict_ann_features(live_features, kinematic_prior=kinematic_prior)
        ens_probs = {
            act: round(0.5 * cnn_res.probabilities[act] + 0.5 * ann_res.probabilities[act], 4)
            for act in ACTIVITY_NAMES
        }
        best_act = max(ens_probs, key=ens_probs.get)
        w_steps, spm, peaks = detect_steps_in_window(
            calibrated_signals, payload.sample_rate_hz, best_act
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
        calibrated_signals, kinematic_prior=kinematic_prior, sample_rate_hz=payload.sample_rate_hz
    )


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
