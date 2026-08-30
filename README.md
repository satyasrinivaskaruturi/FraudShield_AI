# 🛡️ FraudShield AI

## AI-Powered Payment Risk Manager

FraudShield AI is a defense-only payment risk management prototype that combines:

- XGBoost supervised fraud detection
- Isolation Forest anomaly detection
- Fraud-spike detection
- Risk fusion
- Cost-sensitive risk thresholds
- Risk Hold / Verify / Approve policy engine
- AI Voice Verification
- PAN verification workflow
- Safe payment release simulation

## System Workflow

Transaction
↓
XGBoost + Isolation Forest
↓
Fraud-Spike Detection
↓
Risk Fusion
↓
Risk Score
↓
Approve / Verify / Risk Hold
↓
AI Voice Verification
↓
PAN Verification
↓
Release / Keep Hold

## Run the AI Dashboard

Install dependencies:

```bash
pip install -r requirements_voice_fixed.txt

http://localhost:8501




### One important thing

Your screenshot still says:

> **“It does not connect to or control payment rails.”**

Keep that statement. It's actually good for your competition because it clearly establishes that this is a **defensive prototype**, not a system making real financial transactions.

For the final GitHub submission, I would also update the README to include a **system architecture diagram, screenshots of your working dashboard, installation instructions, and a 30-second explanation of the AI model**.
