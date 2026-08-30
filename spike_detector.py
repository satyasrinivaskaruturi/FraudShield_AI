import numpy as np
import pandas as pd

def detect_fraud_spike(scores, window=20, z_threshold=3.0):
    """
    scores: chronological sequence of transaction risk scores.
    Uses a rolling median/MAD baseline. Returns an alert for the latest point.
    """
    s = pd.Series(scores, dtype=float)
    if len(s) < 6:
        return {"spike": False, "spike_score": 0.0, "message": "Insufficient history"}

    window = min(window, len(s) - 1)
    history = s.iloc[-window-1:-1]
    median = float(history.median())
    mad = float((history - median).abs().median())
    robust_z = abs(float(s.iloc[-1]) - median) / (1.4826 * mad + 1e-6)
    spike = robust_z >= z_threshold

    return {
        "spike": bool(spike),
        "spike_score": round(min(1.0, robust_z / (z_threshold * 2)), 4),
        "robust_z": round(robust_z, 3),
        "baseline_median": round(median, 4),
        "message": "FRAUD SPIKE DETECTED" if spike else "No fraud spike detected"
    }
