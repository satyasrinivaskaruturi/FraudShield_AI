import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# FRAUDSHIELD AI - RISK ENGINE
# ============================================================

ARTIFACTS = Path("artifacts")


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

with open(ARTIFACTS / "metadata.json", "r") as f:
    META = json.load(f)


# ------------------------------------------------------------
# Load feature list
# ------------------------------------------------------------

FEATURES = META["features"]


# ------------------------------------------------------------
# Load trained models
# ------------------------------------------------------------

XGB = joblib.load(
    ARTIFACTS / "xgb_fraud.joblib"
)

ISO = joblib.load(
    ARTIFACTS / "isolation_forest.joblib"
)


# ------------------------------------------------------------
# Load decision threshold
# ------------------------------------------------------------

THRESHOLD = float(
    META["metrics"]["threshold"]
)


# ------------------------------------------------------------
# Isolation Forest calibration
# ------------------------------------------------------------

CAL = META["metrics"].get(
    "anomaly_calibration",
    {
        "lo": 0.0,
        "hi": 1.0
    }
)


# ============================================================
# ANOMALY SCORE
# ============================================================

def score_anomaly(X):

    # Isolation Forest returns one value per transaction
    raw = -ISO.decision_function(X)

    # Convert calibration values to normal Python floats
    lo = float(CAL["lo"])
    hi = float(CAL["hi"])

    # Normalize anomaly score between 0 and 1
    score = np.clip(
        (raw - lo) /
        (hi - lo + 1e-9),
        0,
        1
    )

    # Convert NumPy array into one normal Python float
    score = np.asarray(score).reshape(-1)[0]

    return float(score)


# ============================================================
# TRANSACTION RISK SCORING
# ============================================================

def score_transaction(tx):

    # --------------------------------------------------------
    # Create input row
    # --------------------------------------------------------

    row = {
        feature: tx.get(feature, 0)
        for feature in FEATURES
    }

    X = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # XGBoost fraud probability
    # --------------------------------------------------------

    fraud_probability = float(
        XGB.predict_proba(X)[0][1]
    )

    # --------------------------------------------------------
    # Isolation Forest anomaly score
    # --------------------------------------------------------

    anomaly_score = score_anomaly(X)

    # --------------------------------------------------------
    # AI RISK FUSION
    #
    # XGBoost       = 80%
    # Isolation     = 20%
    # --------------------------------------------------------

    risk_score = (
        0.80 * fraud_probability
        +
        0.20 * anomaly_score
    )

    # --------------------------------------------------------
    # DECISION ENGINE
    # --------------------------------------------------------

    if risk_score >= THRESHOLD:

        action = "RISK_HOLD"

    elif risk_score >= max(
        0.35,
        THRESHOLD * 0.55
    ):

        action = "VERIFY"

    else:

        action = "APPROVE"

    # --------------------------------------------------------
    # EXPLAINABLE AI REASONS
    # --------------------------------------------------------

    reasons = []

    if row.get("amount_ratio", 0) > 5:

        reasons.append(
            "Transaction amount is unusually high"
        )

    if row.get("transactions_10min", 0) >= 7:

        reasons.append(
            "High transaction velocity"
        )

    if row.get("transactions_1h", 0) >= 15:

        reasons.append(
            "Unusually high transaction frequency"
        )

    if row.get("new_device", 0) == 1:

        reasons.append(
            "New device detected"
        )

    if row.get("new_location", 0) == 1:

        reasons.append(
            "New location detected"
        )

    if row.get("hour", 12) <= 4:

        reasons.append(
            "Unusual transaction time"
        )

    if row.get("previous_failures", 0) >= 2:

        reasons.append(
            "Multiple previous transaction failures"
        )

    # If nothing suspicious was detected
    if not reasons:

        reasons.append(
            "No dominant behavioral anomaly detected"
        )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "anomaly_score": round(
            anomaly_score,
            4
        ),

        "risk_score": round(
            risk_score,
            4
        ),

        "risk_percent": round(
            risk_score * 100,
            2
        ),

        "action": action,

        "reasons": reasons
    }