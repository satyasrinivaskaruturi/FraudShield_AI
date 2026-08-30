import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

FEATURES = [
    "amount", "hour", "customer_age_days", "previous_transactions",
    "previous_failures", "avg_amount", "amount_ratio",
    "transactions_10min", "transactions_1h", "new_device",
    "new_location", "is_weekend"
]

def make_data(n=30000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    # Start with mostly legitimate behavior.
    fraud_rate = 0.04
    y = rng.binomial(1, fraud_rate, n)

    hour = rng.integers(6, 23, n)
    amount = rng.lognormal(mean=6.8, sigma=0.55, size=n)
    previous_transactions = rng.poisson(28, n) + 1
    previous_failures = rng.poisson(0.35, n)
    avg_amount = rng.lognormal(mean=6.75, sigma=0.45, size=n)
    transactions_10min = rng.poisson(1.0, n)
    transactions_1h = transactions_10min + rng.poisson(2.0, n)
    new_device = rng.binomial(1, 0.05, n)
    new_location = rng.binomial(1, 0.04, n)
    customer_age_days = rng.integers(30, 1800, n)
    is_weekend = (rng.integers(0, 7, n) >= 5).astype(int)

    # Inject a small set of realistic fraud behaviors.
    fraud_idx = np.where(y == 1)[0]
    m = len(fraud_idx)
    hour[fraud_idx] = rng.choice([0, 1, 2, 3, 4, 23], size=m)
    amount[fraud_idx] *= rng.uniform(4, 15, size=m)
    transactions_10min[fraud_idx] += rng.integers(5, 15, size=m)
    transactions_1h[fraud_idx] += rng.integers(8, 30, size=m)
    new_device[fraud_idx] = rng.binomial(1, 0.85, m)
    new_location[fraud_idx] = rng.binomial(1, 0.65, m)
    previous_failures[fraud_idx] += rng.integers(1, 4, size=m)

    amount_ratio = amount / np.maximum(avg_amount, 1)

    df = pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "customer_age_days": customer_age_days,
        "previous_transactions": previous_transactions,
        "previous_failures": previous_failures,
        "avg_amount": avg_amount.round(2),
        "amount_ratio": amount_ratio.round(3),
        "transactions_10min": transactions_10min,
        "transactions_1h": transactions_1h,
        "new_device": new_device,
        "new_location": new_location,
        "is_weekend": is_weekend,
        "fraud": y
    })
    return df

def choose_threshold(y, p, fp_cost=100.0, fn_cost=5000.0, min_recall=0.80):
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        recall = tp / max(tp + fn, 1)
        cost = fp * fp_cost + fn * fn_cost
        rows.append((t, cost, recall, fp, fn))
    valid = [r for r in rows if r[2] >= min_recall]
    best = min(valid or rows, key=lambda r: r[1])
    return best

def main():
    df = make_data()
    X = df[FEATURES]
    y = df["fraud"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        n_jobs=4
    )
    xgb.fit(X_train, y_train)

    # Fit anomaly detector only on legitimate training examples.
    iso = IsolationForest(
        n_estimators=250, contamination=0.03,
        random_state=RANDOM_STATE, n_jobs=4
    )
    iso.fit(X_train[y_train == 0])

    legitimate_train_raw = -iso.decision_function(X_train[y_train == 0])
    anomaly_lo, anomaly_hi = np.percentile(legitimate_train_raw, [1, 99])

    def anomaly_score(Xpart):
        raw = -iso.decision_function(Xpart)
        return np.clip(
            (raw - anomaly_lo) / (anomaly_hi - anomaly_lo + 1e-9), 0, 1
        )

    p_val = xgb.predict_proba(X_val)[:, 1]
    a_val = anomaly_score(X_val)
    # Fusion weight selected for prototype; can be tuned on validation data.
    fusion_val = 0.80 * p_val + 0.20 * a_val

    threshold, cost, val_recall, val_fp, val_fn = choose_threshold(
        y_val, fusion_val, fp_cost=100, fn_cost=5000, min_recall=0.80
    )

    p_test = xgb.predict_proba(X_test)[:, 1]
    a_test = anomaly_score(X_test)
    fusion_test = 0.80 * p_test + 0.20 * a_test
    pred_test = (fusion_test >= threshold).astype(int)

    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)
    pr_auc = average_precision_score(y_test, fusion_test)
    roc_auc = roc_auc_score(y_test, fusion_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0,1]).ravel()
    total_cost = fp * 100 + fn * 5000

    metrics = {
        "dataset_rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "threshold": float(threshold),
        "fusion": {"xgboost_weight": 0.80, "isolation_forest_weight": 0.20},
        "anomaly_calibration": {"lo": float(anomaly_lo), "hi": float(anomaly_hi)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_cost": 100,
        "false_negative_cost": 5000,
        "test_business_cost": float(total_cost),
    }

    joblib.dump(xgb, ARTIFACTS / "xgb_fraud.joblib")
    joblib.dump(iso, ARTIFACTS / "isolation_forest.joblib")
    with open(ARTIFACTS / "metadata.json", "w") as f:
        json.dump({"features": FEATURES, "metrics": metrics}, f, indent=2)

    df.to_csv(ARTIFACTS / "synthetic_transactions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("\nSaved trained models to artifacts/")

if __name__ == "__main__":
    main()
