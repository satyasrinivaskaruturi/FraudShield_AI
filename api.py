from fastapi import FastAPI
from pydantic import BaseModel, Field
from risk_engine import score_transaction
from spike_detector import detect_fraud_spike

app = FastAPI(title="FraudShield AI", version="1.0")

class Transaction(BaseModel):
    amount: float
    hour: int = Field(ge=0, le=23)
    customer_age_days: int
    previous_transactions: int
    previous_failures: int
    avg_amount: float
    amount_ratio: float
    transactions_10min: int
    transactions_1h: int
    new_device: int
    new_location: int
    is_weekend: int

@app.get("/")
def root():
    return {"name": "FraudShield AI", "status": "ready"}

@app.post("/score")
def score(tx: Transaction):
    return score_transaction(tx.model_dump())

@app.post("/spike")
def spike(scores: list[float]):
    return detect_fraud_spike(scores)
