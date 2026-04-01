import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "LR.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
expected_columns = joblib.load(BASE_DIR / "columns.pkl")

# ---------------- UI DESIGN ---------------- #
st.set_page_config(page_title="Heart Risk Predictor", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
    ❤️ Heart Disease Risk Predictor by Pallab
    </h1>
    <p style='text-align: center;'>👈 Enter patient details to analyze risk</p>
    """,
    unsafe_allow_html=True
)

# -------- Sidebar Input -------- #
st.sidebar.header("📝 Patient Details")

age = st.sidebar.slider("Age",18,100,40)
sex = st.sidebar.selectbox("Sex",['M','F'])
chest_pain = st.sidebar.selectbox("Chest Pain",['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.sidebar.number_input("Resting BP", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting BS >120", ['NO','YES'])
resting_ecg = st.sidebar.selectbox('ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 180, 130)
exercise_angina = st.sidebar.selectbox('Exercise Angina', ['YES', 'NO'])
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# -------- Predict Button -------- #
if st.sidebar.button("🔍 Predict Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == 'YES' else 0,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # -------- Result Section -------- #
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Heart Disease Risk", f"{probability*100:.2f}%")

    with col2:
        if probability > 0.7:
            st.error("🔴 High Risk")
        elif probability > 0.4:
            st.warning("🟡 Moderate Risk")
        else:
            st.success("🟢 Low Risk")

    # -------- Progress Bar -------- #
    st.progress(int(probability * 100))

    # -------- Extra Info -------- #
    st.info("This prediction is based on ML analysis of medical data patterns.")

    