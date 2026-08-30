from risk_engine import score_transaction
from spike_detector import detect_fraud_spike

normal = {
    "amount": 850, "hour": 14, "customer_age_days": 420,
    "previous_transactions": 30, "previous_failures": 0,
    "avg_amount": 900, "amount_ratio": 0.94,
    "transactions_10min": 1, "transactions_1h": 3,
    "new_device": 0, "new_location": 0, "is_weekend": 0
}

suspicious = {
    "amount": 42500, "hour": 2, "customer_age_days": 30,
    "previous_transactions": 3, "previous_failures": 3,
    "avg_amount": 1200, "amount_ratio": 35.4,
    "transactions_10min": 10, "transactions_1h": 18,
    "new_device": 1, "new_location": 1, "is_weekend": 0
}

print("NORMAL TRANSACTION")
print(score_transaction(normal))

print("\nSUSPICIOUS TRANSACTION")
print(score_transaction(suspicious))

history = [0.10,0.12,0.09,0.11,0.13,0.10,0.14,0.09,0.12,0.11,
           0.10,0.13,0.12,0.11,0.10,0.14,0.12,0.13,0.11,0.12,
           0.96]
print("\nFRAUD SPIKE")
print(detect_fraud_spike(history))
