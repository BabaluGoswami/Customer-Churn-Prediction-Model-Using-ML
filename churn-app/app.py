import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# -------------------- Load Model Safely --------------------
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

final_results = pd.read_csv(os.path.join(BASE_DIR, "Final_Results.csv"))

st.title("📊 Customer Churn Prediction & Dashboard")

# -------------------- Sidebar --------------------
st.sidebar.header("🔹 Demo Options")
demo_mode = st.sidebar.radio(
    "Select Mode:",
    ["Dashboard (Existing Customers)", "Predict Individual Customer"]
)

# ==================== DASHBOARD ====================
if demo_mode == "Dashboard (Existing Customers)":

    st.header("📈 Customer Churn Dashboard")

    total = len(final_results)
    churn = final_results["Predicted_Churn"].sum()
    stayed = total - churn

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total)
    col2.metric("Churn Customers", churn)
    col3.metric("Stayed Customers", stayed)

    st.subheader("📊 Distribution")

    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(
        [churn, stayed],
        labels=["Churn", "Stayed"],
        autopct="%1.1f%%",
        startangle=90
    )
    st.pyplot(fig)

    acc = (final_results["Predicted_Churn"] == final_results["Actual_Churn"]).mean()
    st.success(f"Model Accuracy: {acc*100:.2f}%")

    st.subheader("📄 Customer Data")

    option = st.radio("View:", ["All", "Churn", "Stayed"])

    if option == "Churn":
        df_show = final_results[final_results["Predicted_Churn"] == 1]
    elif option == "Stayed":
        df_show = final_results[final_results["Predicted_Churn"] == 0]
    else:
        df_show = final_results

    df_show = df_show.copy()
    df_show.index = range(1, len(df_show) + 1)
    st.dataframe(df_show)

# ==================== PREDICTION ====================
else:

    st.header("🤖 Predict Customer Churn")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 5000.0, 100.0)

    if st.button("Predict"):

        data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': backup,
            'DeviceProtection': device,
            'TechSupport': tech,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total_charges
        }

        df_input = pd.DataFrame([data])

        # Yes/No convert
        yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
        for col in yes_no_cols:
            df_input[col] = df_input[col].map({'Yes':1,'No':0})

        # One-hot
        df_input = pd.get_dummies(df_input)

        # 🔥 FIX (IMPORTANT)
        for col in columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[columns]
        df_input = df_input.astype(float)

        # Scale
        scaled = scaler.transform(df_input)

        # Predict
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        # Risk
        if prob > 0.7:
            risk = "High Risk 🔴"
        elif prob > 0.4:
            risk = "Medium Risk 🟡"
        else:
            risk = "Low Risk 🟢"

        st.subheader("Result")

        if pred == 1:
            st.error(f"Churn: YES ❌ | {risk}")
        else:
            st.success(f"Churn: NO ✅ | {risk}")

        st.write(f"Probability: {prob:.2f}")
