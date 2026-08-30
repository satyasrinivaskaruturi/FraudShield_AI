import streamlit as st
from risk_engine import score_transaction
from spike_detector import detect_fraud_spike

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudShield AI")
st.subheader("AI-Powered Payment Risk Manager")
st.write(
    "XGBoost + Isolation Forest + Fraud-Spike Detection + Risk Hold"
)

st.sidebar.header("Payment Details")

amount = st.sidebar.number_input(
    "Transaction Amount (₹)",
    min_value=1.0,
    value=75000.0
)

hour = st.sidebar.slider(
    "Transaction Hour",
    0,
    23,
    2
)

customer_age_days = st.sidebar.number_input(
    "Customer Age (days)",
    min_value=1,
    value=30
)

previous_transactions = st.sidebar.number_input(
    "Previous Transactions",
    min_value=0,
    value=3
)

previous_failures = st.sidebar.number_input(
    "Previous Failed Transactions",
    min_value=0,
    value=3
)

avg_amount = st.sidebar.number_input(
    "Customer Average Amount (₹)",
    min_value=1.0,
    value=1200.0
)

transactions_10min = st.sidebar.number_input(
    "Transactions in Last 10 Minutes",
    min_value=0,
    value=10
)

transactions_1h = st.sidebar.number_input(
    "Transactions in Last 1 Hour",
    min_value=0,
    value=18
)

new_device = st.sidebar.selectbox(
    "New Device?",
    ["Yes", "No"]
)

new_location = st.sidebar.selectbox(
    "New Location?",
    ["Yes", "No"]
)

is_weekend = st.sidebar.selectbox(
    "Weekend?",
    ["Yes", "No"]
)


transaction = {

    "amount": amount,

    "hour": hour,

    "customer_age_days": customer_age_days,

    "previous_transactions":
        previous_transactions,

    "previous_failures":
        previous_failures,

    "avg_amount":
        avg_amount,

    "amount_ratio":
        amount / max(avg_amount, 1),

    "transactions_10min":
        transactions_10min,

    "transactions_1h":
        transactions_1h,

    "new_device":
        int(new_device == "Yes"),

    "new_location":
        int(new_location == "Yes"),

    "is_weekend":
        int(is_weekend == "Yes")
}


st.divider()

st.header("💳 Payment Analysis")

if st.button(
    "🔍 ANALYZE PAYMENT",
    type="primary",
    use_container_width=True
):

    result = score_transaction(transaction)

    st.session_state["result"] = result


if "result" in st.session_state:

    result = st.session_state["result"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fraud Probability",
        f"{result['fraud_probability'] * 100:.1f}%"
    )

    col2.metric(
        "Anomaly Score",
        f"{result['anomaly_score'] * 100:.1f}%"
    )

    col3.metric(
        "Final Risk",
        f"{result['risk_percent']:.1f}%"
    )

    col4.metric(
        "Decision",
        result["action"]
    )

    st.divider()

    if result["action"] == "RISK_HOLD":

        st.error(
            f"🔒 PAYMENT ON RISK HOLD\n\n"
            f"Risk Score: {result['risk_percent']:.1f}%"
        )

        st.warning(
            "Payment settlement is paused until "
            "verification is completed."
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "✅ VERIFY & RELEASE",
                use_container_width=True
            ):

                st.success(
                    "Payment RELEASED successfully."
                )

        with col2:

            if st.button(
                "⛔ REJECT PAYMENT",
                use_container_width=True
            ):

                st.error(
                    "Payment REJECTED."
                )

    elif result["action"] == "VERIFY":

        st.warning(
            f"⚠️ VERIFICATION REQUIRED\n\n"
            f"Risk Score: {result['risk_percent']:.1f}%"
        )

    else:

        st.success(
            f"✅ PAYMENT APPROVED\n\n"
            f"Risk Score: {result['risk_percent']:.1f}%"
        )


    st.subheader("🔎 Why did the AI flag this?")

    for reason in result["reasons"]:

        st.write(
            f"• {reason}"
        )


st.divider()

st.header("📈 Merchant Fraud-Spike Detection")

st.write(
    "Enter chronological risk scores separated by commas."
)

history = st.text_input(
    "Risk Score History",
    "0.10,0.12,0.09,0.11,0.13,0.10,0.14,0.09,0.12,0.11,0.10,0.13,0.12,0.11,0.10,0.14,0.12,0.13,0.11,0.12,0.96"
)

if st.button("📊 DETECT FRAUD SPIKE"):

    try:

        scores = [
            float(x.strip())
            for x in history.split(",")
        ]

        result = detect_fraud_spike(scores)

        if result["spike"]:

            st.error(
                "🚨 FRAUD SPIKE DETECTED"
            )

        else:

            st.success(
                "✅ No significant fraud spike detected."
            )

        st.metric(
            "Spike Score",
            f"{result['spike_score'] * 100:.1f}%"
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


st.divider()

st.caption(
    "FraudShield AI — Competition Prototype | "
    "Payment actions are simulated and do not control real payment rails."
)