import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# -------------------- Safe Path Setup --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- Load Model & Data --------------------
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))
final_results = pd.read_csv(os.path.join(BASE_DIR, "Final_Results.csv"))

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
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

    total_customers = len(final_results)
    churn_customers = int(final_results["Predicted_Churn"].sum())
    stayed_customers = total_customers - churn_customers

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churn Customers", churn_customers)
    col3.metric("Stayed Customers", stayed_customers)

    # -------- Pie Chart (fixed size) --------
    st.subheader("📊 Customer Distribution")
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(
        [churn_customers, stayed_customers],
        labels=["Churn","Stayed"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Churn vs Stayed")
    st.pyplot(fig)

    # -------- Accuracy --------
    correct = sum(final_results["Predicted_Churn"] == final_results["Actual_Churn"])
    accuracy = correct / total_customers

    st.markdown(
        f"### 🎯 Model Accuracy: `{accuracy*100:.2f}%`"
    )

    # -------- Data View --------
    st.subheader("📄 Customer Data")

    option = st.radio("View:", ["All", "Churn", "Stayed"])

    if option == "All":
        df_display = final_results.copy()
    elif option == "Churn":
        df_display = final_results[final_results["Predicted_Churn"] == 1]
    else:
        df_display = final_results[final_results["Predicted_Churn"] == 0]

    df_display = df_display.reset_index(drop=True)
    df_display.index += 1

    st.dataframe(df_display)

# ==================== INDIVIDUAL ====================
else:

    st.header("🤖 Predict Individual Customer")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
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
    total = st.number_input("Total Charges", 0.0, 5000.0, 100.0)

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
            'TotalCharges': total
        }

        df_input = pd.DataFrame([data])

        # Yes/No encoding
        for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
            df_input[col] = df_input[col].map({'Yes':1,'No':0})

        # One hot
        df_input = pd.get_dummies(df_input)

        # -------- FIX: column mismatch --------
        df_input = df_input.reindex(columns=columns, fill_value=0)

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

        # Result
        st.subheader("✅ Result")

        if pred == 1:
            st.error(f"Churn: YES ❌ | {risk}")
        else:
            st.success(f"Churn: NO ✅ | {risk}")

        st.write(f"Probability: {prob:.2f}")

        # Recommendation
        st.subheader("💡 Recommendation")
        if prob > 0.7:
            st.write("Give discount / call customer")
        elif prob > 0.4:
            st.write("Send offers")
        else:
            st.write("No action needed")
