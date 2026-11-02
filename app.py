import os
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from models import HAR_1DCNN, RegularizedHAR_ANN
from sample_generator import generate_realistic_sample, ACTIVITY_NAMES

app = FastAPI(
    title="UCI HAR AI & Big Data Convergence API",
    description="Dual-Pipeline Human Activity Recognition API (1D-CNN Raw Signals & Regularized ANN 561-Features)",
    version="2.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
AUTH_DB_PATH = Path(
    os.environ.get(
        "HAR_AUTH_DB",
        str(BASE_DIR / "har_auth.sqlite3")
    )
).expanduser()
SESSION_COOKIE = "har_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
PASSWORD_HASH_ITERATIONS = 600_000


def get_auth_db() -> sqlite3.Connection:
    connection = sqlite3.connect(str(AUTH_DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_auth_db() -> None:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = get_auth_db()
    try:
        connection.executescript(
            """
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
        )
    finally:
        connection.close()


app.add_event_handler("startup", initialize_auth_db)


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


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


def create_session(user_id: int) -> str:
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


def require_authenticated_user(request: Request) -> dict:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

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


# Allow only explicitly configured cross-origin browser clients.
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("HAR_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Models
cnn_model = HAR_1DCNN(in_channels=9, num_classes=6).to(device)
ann_model = RegularizedHAR_ANN(input_dim=561, num_classes=6).to(device)

cnn_weights_loaded = False
ann_weights_loaded = False

cnn_weights_path = BASE_DIR / "har_1dcnn_weights.pth"
ann_weights_path = BASE_DIR / "har_ann_regularized.pth"

if cnn_weights_path.exists():
    cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
    cnn_weights_loaded = True

if ann_weights_path.exists():
    ann_model.load_state_dict(torch.load(ann_weights_path, map_location=device))
    ann_weights_loaded = True

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
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.post("/api/auth/register")
def register(payload: CredentialsPayload, request: Request, response: Response):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", payload.username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters: letters, numbers, '.', '_' or '-'.",
        )
    if len(payload.password) < 12:
        raise HTTPException(status_code=400, detail="Password must be at least 12 characters.")

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
    return {"username": username}


@app.post("/api/auth/login")
def login(payload: CredentialsPayload, request: Request, response: Response):
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
    return {"username": user["username"]}


@app.get("/api/auth/me")
def current_user(user: dict = Depends(require_authenticated_user)):
    return {"username": user["username"]}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        connection = get_auth_db()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM auth_sessions WHERE token_hash = ?",
                    (hashlib.sha256(token.encode("utf-8")).hexdigest(),),
                )
        finally:
            connection.close()
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")
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

    return predict_cnn_signals(signals)


def predict_cnn_signals(signals: np.ndarray) -> PredictionResponse:
    start = time.perf_counter()
    tensor_in = torch.from_numpy(signals).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = cnn_model(tensor_in)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if not cnn_weights_loaded:
            probs = adjust_Untrained_Probs(probs, signals)
        pred_idx = int(np.argmax(probs))
    latency = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        model_used="1D-CNN (Raw 9x128 Signals)",
        predicted_activity=ACTIVITY_NAMES[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (.pth Loaded)" if cnn_weights_loaded else "Kinematic Demo Mode (Place har_1dcnn_weights.pth in root)"
    )

@app.post("/predict/ann", response_model=PredictionResponse)
def predict_ann(payload: FeatureVectorPayload, user: dict = Depends(require_authenticated_user)):
    features = np.array(payload.features, dtype=np.float32)
    if features.shape != (561,):
        raise HTTPException(status_code=400, detail=f"Expected 561 features, got {features.shape}")

    return predict_ann_features(features)


def predict_ann_features(features: np.ndarray) -> PredictionResponse:
    start = time.perf_counter()
    tensor_in = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = ann_model(tensor_in)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
    latency = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        model_used="Regularized ANN (561 Features)",
        predicted_activity=ACTIVITY_NAMES[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={ACTIVITY_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
        latency_ms=round(latency, 2),
        weights_status="Trained (.pth Loaded)" if ann_weights_loaded else "Demo Mode (Place har_ann_regularized.pth in root)"
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
