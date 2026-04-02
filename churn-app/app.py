# 📁 Customer Churn Streamlit Dashboard - Enhanced & Attractive Version
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Load Model & Data --------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load final results
final_results = pd.read_csv("Final_Results.csv")

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Page Title --------------------
st.markdown("<h1 style='text-align:center;color:#4B7BEC'>📊 Customer Churn Dashboard</h1>", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.header("🔹 Demo Options")
demo_mode = st.sidebar.radio("Select Mode:", ["Dashboard (Existing Customers)", "Predict Individual Customer"])

# -------------------- Dashboard Mode --------------------
if demo_mode == "Dashboard (Existing Customers)":
    st.header("📈 Customer Churn Overview")
    
    total_customers = len(final_results)
    churn_customers = final_results["Predicted_Churn"].sum()
    stayed_customers = total_customers - churn_customers

    # ---------------- Metrics ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers, "👥")
    col2.metric("Predicted Churn", churn_customers, "⚠️")
    col3.metric("Predicted Stayed", stayed_customers, "✅")

    # ---------------- Pie Chart ----------------
    st.subheader("📊 Customer Distribution")
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ["#FF4B4B", "#4BCB6B"]
    wedges, texts, autotexts = ax.pie(
        [churn_customers, stayed_customers],
        labels=["Churn","Stayed"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05,0),
        shadow=True,
        wedgeprops={"edgecolor":"black"}
    )
    for text in texts+autotexts:
        text.set_fontsize(12)
    ax.axis('equal')
    ax.set_title("Churn vs Stayed Customers", fontsize=16)
    st.pyplot(fig)

    # ---------------- Model Accuracy ----------------
    correct_predictions = sum(final_results["Predicted_Churn"] == final_results["Actual_Churn"])
    accuracy = correct_predictions / total_customers
    st.markdown(f"<h3 style='color:#4B7BEC'>Model Accuracy: {accuracy*100:.2f}%</h3>", unsafe_allow_html=True)

    # ---------------- Customer Table ----------------
    st.subheader("📄 Customer Data")
    view_option = st.radio("View Customers:", ["All", "Churn", "Stayed"])
    if view_option == "All":
        df_display = final_results.copy()
    elif view_option == "Churn":
        df_display = final_results[final_results["Predicted_Churn"]==1].copy()
    else:
        df_display = final_results[final_results["Predicted_Churn"]==0].copy()
    df_display.index = range(1, len(df_display)+1)
    st.dataframe(df_display, use_container_width=True)

# -------------------- Individual Prediction Mode --------------------
else:
    st.header("🤖 Predict Individual Customer Churn")
    
    # ---------------- Input Form ----------------
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0,1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with col2:
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
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
            total = st.number_input("Total Charges", min_value=0.0, max_value=5000.0, value=100.0)
        submit = st.form_submit_button("Predict")

    # ---------------- Prediction Logic ----------------
    if submit:
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

        # Encode Yes/No columns
        yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
        for col in yes_no_cols:
            df_input[col] = df_input[col].map({'Yes':1,'No':0})

        # One-hot encode categorical
        df_input = pd.get_dummies(df_input)
        for col in columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[columns]

        # Scale
        scaled_input = scaler.transform(df_input)

        # Predict
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # Risk level
        if prob > 0.7:
            risk = "High Risk 🔴"
        elif prob > 0.4:
            risk = "Medium Risk 🟡"
        else:
            risk = "Low Risk 🟢"

        # ---------------- Result Display ----------------
        st.markdown("### ✅ Prediction Result")
        if pred == 1:
            st.error(f"Predicted Churn: YES ❌ | {risk}")
        else:
            st.success(f"Predicted Churn: NO ✅ | {risk}")
        st.info(f"Churn Probability: {prob:.2f}")

        # Recommendations
        st.markdown("### 💡 Recommendation")
        if prob > 0.7:
            st.write("High Risk → Offer discount / retention call")
        elif prob > 0.4:
            st.write("Medium Risk → Send promotional offers")
        else:
            st.write("Low Risk → No action needed")