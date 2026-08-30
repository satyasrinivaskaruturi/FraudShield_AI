# FraudShield AI — Unified AI Risk Manager

A defense-only prototype combining:
- XGBoost supervised fraud detection
- Isolation Forest anomaly detection
- Fraud-spike detection using rolling robust statistics
- Risk fusion
- Cost-sensitive threshold selection
- Risk Hold / Verify / Approve policy engine
- SHAP-ready feature importance through XGBoost gain importance

## Run

```bash
pip install -r requirements.txt
python train.py
python demo.py
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000/docs

## Important
This prototype uses synthetic data and a simulated payment state machine.
It does not connect to or control real payment rails.
