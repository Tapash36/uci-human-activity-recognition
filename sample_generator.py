import numpy as np
import zlib

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

def generate_realistic_sample(activity_name: str):
    """Generates realistic 9x128 inertial signals and 561 feature vectors for testing/demo."""
    rng = np.random.default_rng(zlib.crc32(activity_name.encode("utf-8")))
    t = np.linspace(0, 2.56, 128)
    signals = np.zeros((9, 128), dtype=np.float32)
    
    if activity_name == "WALKING":
        freq = 2.0  # 2 steps per second
        signals[0] = 0.45 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # body_acc_x
        signals[1] = 0.25 * np.cos(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # body_acc_y
        signals[2] = 0.30 * np.sin(4 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # body_acc_z
        signals[3] = 0.60 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # gyro_x
        signals[4] = 0.40 * np.cos(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # gyro_y
        signals[5] = 0.35 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(128)  # gyro_z
        signals[6] = signals[0] + 0.98  # total_acc_x (gravity vertical)
        signals[7] = signals[1] + 0.02
        signals[8] = signals[2] + 0.10
    elif activity_name == "WALKING_UPSTAIRS":
        freq = 1.6
        signals[0] = 0.65 * np.sin(2 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[1] = 0.35 * np.cos(2 * np.pi * freq * t) + 0.06 * rng.standard_normal(128)
        signals[2] = 0.45 * np.sin(2 * np.pi * freq * t + 0.5) + 0.06 * rng.standard_normal(128)
        signals[3] = 0.85 * np.cos(2 * np.pi * freq * t) + 0.07 * rng.standard_normal(128)
        signals[4] = 0.50 * np.sin(2 * np.pi * freq * t) + 0.06 * rng.standard_normal(128)
        signals[5] = 0.40 * np.cos(2 * np.pi * freq * t) + 0.06 * rng.standard_normal(128)
        signals[6] = signals[0] + 0.90
        signals[7] = signals[1] + 0.25
        signals[8] = signals[2] + 0.30
    elif activity_name == "WALKING_DOWNSTAIRS":
        freq = 2.3
        signals[0] = 0.80 * np.sin(2 * np.pi * freq * t) + 0.12 * rng.standard_normal(128)
        signals[1] = 0.40 * np.sin(4 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[2] = 0.50 * np.cos(2 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[3] = 0.95 * np.sin(2 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[4] = 0.60 * np.cos(2 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[5] = 0.55 * np.sin(2 * np.pi * freq * t) + 0.08 * rng.standard_normal(128)
        signals[6] = signals[0] + 0.92
        signals[7] = signals[1] - 0.15
        signals[8] = signals[2] + 0.20
    elif activity_name == "SITTING":
        signals[0:6] = 0.012 * rng.standard_normal((6, 128))
        signals[6] = 0.88 + 0.01 * rng.standard_normal(128)
        signals[7] = -0.22 + 0.01 * rng.standard_normal(128)
        signals[8] = 0.40 + 0.01 * rng.standard_normal(128)
    elif activity_name == "STANDING":
        signals[0:6] = 0.015 * rng.standard_normal((6, 128))
        signals[6] = 0.99 + 0.01 * rng.standard_normal(128)
        signals[7] = -0.05 + 0.01 * rng.standard_normal(128)
        signals[8] = -0.08 + 0.01 * rng.standard_normal(128)
    else:  # LAYING
        signals[0:6] = 0.008 * rng.standard_normal((6, 128))
        signals[6] = 0.05 + 0.01 * rng.standard_normal(128)
        signals[7] = 0.72 + 0.01 * rng.standard_normal(128)
        signals[8] = 0.68 + 0.01 * rng.standard_normal(128)
        
    features = np.clip(rng.uniform(-0.8, 0.8, size=(561,)), -1.0, 1.0).astype(np.float32)
    return signals.tolist(), features.tolist()
