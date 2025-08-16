import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection Prediction APP")
st.markdown("Please enter the transaction details and use the predict button")

st.divider()

# Input fields
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalance_org = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalance_orig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalance_dest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalance_dest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict"):
    # Raw input
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalance_org": oldbalance_org,
        "newbalance_orig": newbalance_orig,
        "oldbalance_dest": oldbalance_dest,
        "newbalance_dest": newbalance_dest
    }])

    # -------- Feature Engineering (same as training) --------
    input_data["balance_diff_orig"] = input_data["oldbalance_org"] - input_data["amount"]
    input_data["amount_to_newbalance_dest"] = input_data["newbalance_dest"] - input_data["oldbalance_dest"]

    # Business logic (adapt as per your training dataset)
    input_data["is_internal"] = (input_data["oldbalance_org"] > 0) & (input_data["oldbalance_dest"] > 0)
    input_data["is_internal"] = input_data["is_internal"].astype(int)

    # If you had this column in training, fill with 0 unless you expose it in UI
    input_data["is_flagged_fraud"] = 0  

    # --------------------------------------------------------

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction: {int(prediction)}")

    if prediction == 1:
        st.error("🚨 This transaction may be FRAUDULENT!")
    else:
        st.success("✅ This transaction looks safe.")
