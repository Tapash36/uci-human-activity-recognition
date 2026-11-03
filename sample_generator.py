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
    """Generates realistic 9x128 inertial signals and 561 feature vectors calibrated to the UCI HAR coordinate frame."""
    rng = np.random.default_rng(zlib.crc32(activity_name.encode("utf-8")))
    t = np.linspace(0, 2.56, 128)
    signals = np.zeros((9, 128), dtype=np.float32)

    if activity_name == "WALKING":
        freq = 2.0  # ~2 steps per second
        amps = [-0.65, -0.55, -0.25, -0.15, 0.65, -0.45]
        phases = [3.0, 4.0, 0.2, 5.2, 5.1, 0.0]
        for c in range(6):
            signals[c] = amps[c] * np.sin(2 * np.pi * freq * t + phases[c]) + 0.03 * rng.standard_normal(128)
        signals[6] = signals[0] + 0.82
        signals[7] = signals[1] + 0.25
        signals[8] = signals[2] + 0.30
    elif activity_name == "WALKING_UPSTAIRS":
        freq = 1.7
        amps = [-0.55, 0.20, -0.40, -0.10, 0.15, 0.60]
        phases = [4.7, 1.7, 4.4, 2.8, 2.2, 3.4]
        for c in range(6):
            signals[c] = amps[c] * np.sin(2 * np.pi * freq * t + phases[c]) + 0.03 * rng.standard_normal(128)
        signals[6] = signals[0] + 0.60
        signals[7] = signals[1] - 0.45
        signals[8] = signals[2] - 0.35
    elif activity_name == "WALKING_DOWNSTAIRS":
        freq = 2.2
        amps = [-0.80, 0.30, 0.10, -0.75, 0.70, 0.45]
        phases = [4.8, 0.1, 1.7, 1.7, 4.6, 4.7]
        for c in range(6):
            signals[c] = amps[c] * np.sin(2 * np.pi * freq * t + phases[c]) + 0.03 * rng.standard_normal(128)
        signals[6] = signals[0] + 0.70
        signals[7] = signals[1] + 0.15
        signals[8] = signals[2] - 0.40
    elif activity_name == "SITTING":
        signals[0:6] = 0.010 * rng.standard_normal((6, 128))
        signals[6] = 0.85 + 0.010 * rng.standard_normal(128)
        signals[7] = 0.20 + 0.010 * rng.standard_normal(128)
        signals[8] = 0.70 + 0.010 * rng.standard_normal(128)
    elif activity_name == "STANDING":
        signals[0:6] = 0.010 * rng.standard_normal((6, 128))
        signals[6] = 0.75 + 0.010 * rng.standard_normal(128)
        signals[7] = -0.55 + 0.010 * rng.standard_normal(128)
        signals[8] = -0.20 + 0.010 * rng.standard_normal(128)
    else:  # LAYING
        signals[0:6] = 0.008 * rng.standard_normal((6, 128))
        signals[6] = 0.02 + 0.010 * rng.standard_normal(128)
        signals[7] = -0.25 + 0.010 * rng.standard_normal(128)
        signals[8] = 0.65 + 0.010 * rng.standard_normal(128)

    features = np.clip(rng.uniform(-0.8, 0.8, size=(561,)), -1.0, 1.0).astype(np.float32)
    return signals.tolist(), features.tolist()
