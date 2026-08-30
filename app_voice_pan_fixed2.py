import re
import io
import streamlit as st
from risk_engine import score_transaction
from spike_detector import detect_fraud_spike

VOICE_AVAILABLE = hasattr(st, "audio_input")

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield AI")
st.subheader("AI-Powered Payment Risk Manager")
st.write("XGBoost + Isolation Forest + Fraud-Spike Detection + Risk Hold + Voice Verification")

for k, v in {
    "result": None, "verification_started": False, "verification_step": 0,
    "answers": {}, "pan_result": None, "release_authorized": False
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------- PAYMENT INPUT ----------
st.sidebar.header("Payment Details")
amount = st.sidebar.number_input("Transaction Amount (₹)", min_value=1.0, value=75000.0)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 2)
customer_age_days = st.sidebar.number_input("Customer Age (days)", min_value=1, value=30)
previous_transactions = st.sidebar.number_input("Previous Transactions", min_value=0, value=3)
previous_failures = st.sidebar.number_input("Previous Failed Transactions", min_value=0, value=3)
avg_amount = st.sidebar.number_input("Customer Average Amount (₹)", min_value=1.0, value=1200.0)
transactions_10min = st.sidebar.number_input("Transactions in Last 10 Minutes", min_value=0, value=10)
transactions_1h = st.sidebar.number_input("Transactions in Last 1 Hour", min_value=0, value=18)
new_device = st.sidebar.selectbox("New Device?", ["Yes", "No"])
new_location = st.sidebar.selectbox("New Location?", ["Yes", "No"])
is_weekend = st.sidebar.selectbox("Weekend?", ["Yes", "No"])

transaction = {
    "amount": amount, "hour": hour, "customer_age_days": customer_age_days,
    "previous_transactions": previous_transactions, "previous_failures": previous_failures,
    "avg_amount": avg_amount, "amount_ratio": amount / max(avg_amount, 1),
    "transactions_10min": transactions_10min, "transactions_1h": transactions_1h,
    "new_device": int(new_device == "Yes"), "new_location": int(new_location == "Yes"),
    "is_weekend": int(is_weekend == "Yes")
}

QUESTIONS = [
    "What is the purpose of this payment?",
    "Do you recognize the recipient or merchant?",
    "Did you personally initiate this payment?"
]

def reset_verification():
    for k, v in {"verification_started": False, "verification_step": 0,
                 "answers": {}, "pan_result": None, "release_authorized": False}.items():
        st.session_state[k] = v
    for i in range(len(QUESTIONS)):
        st.session_state.pop(f"transcript_{i}", None)

def verify_pan_demo(pan):
    """Prototype-only syntax check. No official KYC/PAN service is contacted."""
    pan = pan.strip().upper()
    valid = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))
    return {"verified": valid, "message":
            "PAN format is valid. Connect an authorized PAN/KYC provider for actual identity verification."
            if valid else "PAN format is invalid."}

def speak(text):
    if not TTS_AVAILABLE: return
    try:
        gTTS(text=text, lang="en").save("voice_response.mp3")
        with open("voice_response.mp3", "rb") as f: st.audio(f.read(), format="audio/mp3")
    except Exception: pass

def speech_to_text(audio_data):
    """Transcribe WAV audio returned by Streamlit's native audio_input."""
    if not SPEECH_AVAILABLE or not audio_data:
        return None
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_data) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        return None

# ---------- EXISTING AI ----------
st.divider(); st.header("💳 Payment Analysis")
if st.button("🔍 ANALYZE PAYMENT", type="primary", use_container_width=True):
    st.session_state.result = score_transaction(transaction)
    reset_verification()

if st.session_state.result:
    result = st.session_state.result
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
    c2.metric("Anomaly Score", f"{result['anomaly_score']*100:.1f}%")
    c3.metric("Final Risk", f"{result['risk_percent']:.1f}%")
    c4.metric("Decision", result["action"])
    st.divider()

    if result["action"] == "RISK_HOLD":
        st.error(f"🔒 PAYMENT ON RISK HOLD\n\nRisk Score: {result['risk_percent']:.1f}%")
        st.warning("Payment settlement is paused in this prototype until verification is completed.")
        st.subheader("🎙️ AI Voice Verification")

        if not st.session_state.verification_started:
            st.write("The assistant collects safe transaction-context information. Never provide OTPs, PINs, CVVs, passwords, or banking credentials.")
            if st.button("🎙️ START VOICE VERIFICATION", type="primary", use_container_width=True):
                st.session_state.verification_started = True
                st.rerun()
        else:
            step = st.session_state.verification_step
            if step < len(QUESTIONS):
                q = QUESTIONS[step]
                st.info("🤖 Assistant: " + q)
                speak(q)
                answer = st.text_input("Type answer (fallback):", key=f"answer_input_{step}")
                saved_transcript = st.session_state.get(f"transcript_{step}", "")
                if saved_transcript and not answer:
                    answer = saved_transcript
                    st.caption("Using latest voice transcript: " + saved_transcript)
                if VOICE_AVAILABLE:
                    st.markdown("**🎙️ Speak your answer:**")
                    audio = st.audio_input(
                        "Click the microphone, speak, then stop recording",
                        key=f"voice_{step}"
                    )
                    if audio is not None:
                        st.audio(audio, format="audio/wav")
                        transcript = speech_to_text(audio)
                        if transcript:
                            st.success("📝 Transcript: " + transcript)
                            # Keep transcript in a separate state variable.
                            # Do not modify the text_input widget state after it is created.
                            st.session_state[f"transcript_{step}"] = transcript
                            answer = transcript
                        else:
                            st.warning(
                                "The recording was received, but speech could not be "
                                "recognized. You can type the answer below."
                            )
                else:
                    st.warning(
                        "Microphone input is unavailable. Upgrade Streamlit with: "
                        "pip install --upgrade streamlit"
                    )
                if st.button("➡️ SAVE & CONTINUE", key=f"next_{step}"):
                    if answer.strip():
                        st.session_state.answers[step] = answer.strip()
                        st.session_state.verification_step += 1
                        st.rerun()
                    st.warning("Please provide an answer.")
            else:
                st.success("✅ Transaction questions completed.")
                st.markdown("### 🪪 PAN Verification")
                st.caption("Demo: checks PAN format only. Actual identity verification requires an authorized PAN/KYC provider.")
                pan = st.text_input("Enter PAN for demo verification", type="password", max_chars=10, placeholder="ABCDE1234F")
                if st.button("🔎 VERIFY PAN", type="primary"):
                    st.session_state.pan_result = verify_pan_demo(pan)
                if st.session_state.pan_result:
                    pr = st.session_state.pan_result
                    if pr["verified"]:
                        st.success("✅ PAN FORMAT VERIFIED")
                        st.info(pr["message"])
                        st.session_state.release_authorized = True
                    else:
                        st.error("❌ PAN VERIFICATION FAILED")
                        st.warning("Payment remains on Risk Hold.")
                        st.session_state.release_authorized = False
                if st.session_state.release_authorized:
                    st.success("🎯 VERIFICATION COMPLETE — Eligible for manual release.")
                    st.warning("Prototype only: release is simulated and does not control real payment rails.")
                    c1,c2 = st.columns(2)
                    with c1:
                        if st.button("✅ VERIFY & RELEASE", use_container_width=True): st.success("Payment RELEASED successfully (simulation).")
                    with c2:
                        if st.button("⛔ REJECT PAYMENT", use_container_width=True):
                            st.session_state.release_authorized = False; st.error("Payment REJECTED.")
                if st.button("🔄 RESTART VERIFICATION"):
                    reset_verification(); st.rerun()

    elif result["action"] == "VERIFY":
        st.warning(f"⚠️ VERIFICATION REQUIRED\n\nRisk Score: {result['risk_percent']:.1f}%")
    else:
        st.success(f"✅ PAYMENT APPROVED\n\nRisk Score: {result['risk_percent']:.1f}%")

    st.subheader("🔎 Why did the AI flag this?")
    for reason in result["reasons"]: st.write("• " + reason)

# ---------- EXISTING SPIKE DETECTOR ----------
st.divider(); st.header("📈 Merchant Fraud-Spike Detection")
history = st.text_input("Risk Score History", "0.10,0.12,0.09,0.11,0.13,0.10,0.14,0.09,0.12,0.11,0.10,0.13,0.12,0.11,0.10,0.14,0.12,0.13,0.11,0.12,0.96")
if st.button("📊 DETECT FRAUD SPIKE"):
    try:
        r = detect_fraud_spike([float(x.strip()) for x in history.split(",")])
        (st.error if r["spike"] else st.success)("🚨 FRAUD SPIKE DETECTED" if r["spike"] else "✅ No significant fraud spike detected.")
        st.metric("Spike Score", f"{r['spike_score']*100:.1f}%")
    except Exception as e: st.error(f"Error: {e}")

st.divider()
st.caption("FraudShield AI — Competition Prototype | Payment actions are simulated and do not control real payment rails.")
