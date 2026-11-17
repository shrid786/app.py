
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="DIABETES PREDICTOR",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #17365D;
    }

    .subtitle {
        color: #64748B;
        font-size: 16px;
    }

    div.stButton > button {
        background-color: #176B87;
        color: white;
        border-radius: 10px;
        height: 48px;
        font-weight: bold;
        width: 100%;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #12566D;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_rf_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🩺 KNOW YOUR RISK")
    st.caption("SCREENING PLATFORM")

    st.divider()

    st.subheader("ABOUT THIS APPLICATION")

    st.write(
        "This application uses a Random Forest "
        "Classifier to estimate diabetes risk "
        "from four input features."
    )
    )

    st.divider()
    st.caption("Developed using Python")

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered diabetes risk screening dashboard</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- INPUT FORM ----------------
st.subheader("👤 Patient Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=500.0,
            value=148.0,
            step=1.0
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=0.0,
            max_value=100.0,
            value=33.6,
            step=0.1
        )

    with col2:
        insulin = st.number_input(
            "Insulin (µU/mL)",
            min_value=0.0,
            max_value=2000.0,
            value=125.0,
            step=1.0
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50
        )

    st.caption(
        "Enter measurements using the same units "
        "and feature definitions used during model training."
    )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk"
    )

# ---------------- PREDICTION ----------------
if submitted:

    patient_data = pd.DataFrame(
        [[glucose, bmi, insulin, age]],
        columns=["Glucose", "BMI", "Insulin", "Age"]
    )

    prediction = model.predict(patient_data)[0]

    probabilities = model.predict_proba(patient_data)[0]

    positive_index = list(model.classes_).index(1)
    probability = probabilities[positive_index] * 100

    st.divider()

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Prediction",
            "High Risk" if prediction == 1 else "Low Risk"
        )

    with col2:
        st.metric(
            "Predicted Class 1 Probability",
            f"{probability:.2f}%"
        )

    st.write("### Probability Visualization")

    st.progress(int(round(probability)))

    st.caption(
        "The probability is the model's estimated "
        "probability for class 1, not a clinically "
        "validated probability of developing diabetes."
    )

    if prediction == 1:

        st.error(
            "⚠️ The model classified this input as "
            "High Risk."
        )

    else:

        st.success(
            "✅ The model classified this input as "
            "Low Risk."
        )

    # ---------------- INPUT SUMMARY ----------------

    st.write("### Patient Input Summary")

    summary = pd.DataFrame({
        "Parameter": ["Glucose", "BMI", "Insulin", "Age"],
        "Value": [glucose, bmi, insulin, age]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ---------------- DOWNLOAD REPORT ----------------

    report = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi],
        "Insulin": [insulin],
        "Age": [age],
        "Prediction": [
            "High Risk" if prediction == 1 else "Low Risk"
        ],
        "Class_1_Probability": [round(probability, 2)]
    })

    st.download_button(
        "📥 Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="diabetes_prediction_report.csv",
        mime="text/csv"
    )

    st.warning(
        "This model is for educational screening only. "
        "A low-risk result does not rule out diabetes, "
        "and a high-risk result does not confirm it. "
        "Consult a qualified healthcare professional."
    )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Diabetes AI Predictor | Built with Python, "
    "Streamlit, Pandas and Random Forest"
)
