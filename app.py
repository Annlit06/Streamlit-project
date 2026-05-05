import streamlit as st
import pandas as pd
import joblib

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Bank Prediction",
    page_icon="💰",
    layout="wide"
)

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("bank_model.pkl")
columns = joblib.load("columns.pkl")

# --------------------------
# CUSTOM CSS (UI GLOW-UP)
# --------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00ADB5;
    text-align: center;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #007B83;
}
.card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# TITLE
# --------------------------
st.title("💰 Bank Subscription Predictor")

st.markdown("### Fill in customer details")

# --------------------------
# LAYOUT (2 COLUMNS)
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Personal Info")
    age = st.slider("Age", 18, 100, 30)
    balance = st.number_input("Balance", value=0)
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

with col2:
    st.markdown("#### 📞 Campaign Info")
    duration = st.number_input("Call Duration", value=100)
    campaign = st.number_input("Campaign", value=1)
    pdays = st.number_input("Pdays", value=0)
    previous = st.number_input("Previous", value=0)

# --------------------------
# LOAN DETAILS
# --------------------------
st.markdown("#### 💳 Financial Status")
col3, col4, col5 = st.columns(3)

with col3:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
with col4:
    loan = st.selectbox("Personal Loan", ["yes", "no"])
with col5:
    default = st.selectbox("Default", ["yes", "no"])

# --------------------------
# INPUT DATA
# --------------------------
input_data = {
    'age': age,
    'balance': balance,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'housing': 1 if housing == "yes" else 0,
    'loan': 1 if loan == "yes" else 0,
    'default': 1 if default == "yes" else 0,
    'education': {"unknown":0,"primary":1,"secondary":2,"tertiary":3}[education]
}

df_input = pd.DataFrame([input_data])
df_input = df_input.reindex(columns=columns, fill_value=0)

# --------------------------
# PREDICTION BUTTON
# --------------------------
st.markdown("---")

if st.button("🔍 Predict Now"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.markdown("## 📊 Result")

    if prediction == 1:
        st.success(f"✅ Customer WILL subscribe ({probability*100:.2f}% confidence)")
    else:
        st.error(f"❌ Customer will NOT subscribe ({(1-probability)*100:.2f}% confidence)")

    # Progress bar
    st.progress(int(probability * 100))