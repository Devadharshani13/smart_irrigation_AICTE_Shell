import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model_data = joblib.load("Farm_Irrigation_System.pkl")
model = model_data['model']

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Smart Sprinkler System", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size:36px;
        font-weight:bold;
        color:#4CAF50;
        text-align:center;
    }
    .sub-title {
        font-size:18px;
        color:#555;
        text-align:center;
    }
    .sprinkler-result {
        padding: 8px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚿 Smart Farm Sprinkler Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Adjust the 20 sensor inputs and check which sprinklers will turn ON/OFF</div>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# Sidebar Sensor Input
# --------------------------
st.sidebar.header("Sensor Inputs (Scaled 0 - 1)")
sensor_values = []
for i in range(20):
    val = st.sidebar.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_values.append(val)

# --------------------------
# Predict Button
# --------------------------
if st.button("💡 Predict Sprinkler Status"):
    with st.spinner("Analyzing Sensor Data..."):
        input_array = np.array(sensor_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

    st.success("Prediction complete!")
    st.markdown("### 🌱 Sprinkler Activation Status:")

    # Display results in 3 columns
    cols = st.columns(3)
    for i, status in enumerate(prediction):
        col = cols[i % 3]
        color = "#4CAF50" if status == 1 else "#F44336"
        label = "ON" if status == 1 else "OFF"
        html_code = f'<div class="sprinkler-result" style="background-color:{color}">Sprinkler {i} (parcel_{i}): {label}</div>'
        col.markdown(html_code, unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Developed as part of Week 3 - AICTE Internship Project", unsafe_allow_html=True)
