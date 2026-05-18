import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 Diabetes ML App")

st.sidebar.info(
    """
    This AI system predicts whether a person is diabetic or not
    using Machine Learning.
    """
)

# Accuracy display
st.sidebar.success("Model Accuracy: 78%")

st.sidebar.success("ROC-AUC Score: 0.84")

# Main Title
st.title("🩺 Diabetes Prediction System")

st.write("Enter patient health details below:")

# Input Fields
pregnancies = st.number_input("Pregnancies", min_value=0.0)

glucose = st.number_input("Glucose Level", min_value=0.0)

blood_pressure = st.number_input("Blood Pressure", min_value=0.0)

skin_thickness = st.number_input("Skin Thickness", min_value=0.0)

insulin = st.number_input("Insulin Level", min_value=0.0)

bmi = st.number_input("BMI", min_value=0.0)

if bmi < 18.5:
    st.info("BMI Status: Underweight")

elif bmi < 25:
    st.success("BMI Status: Normal Weight")

elif bmi < 30:
    st.warning("BMI Status: Overweight")

else:
    st.error("BMI Status: Obese")

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=0.0)

if st.button("Predict Diabetes"):

    
    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            dpf,
                            age]])

    
    std_data = scaler.transform(input_data)

    
    prediction = model.predict(std_data)


    probability = model.predict_proba(std_data)

    confidence = probability[0][1] * 100


    if prediction[0] == 0:

        st.success("✅ The person is NOT diabetic")

        st.info(f"Confidence Score: {100 - confidence:.2f}%")

    else:

        st.error("⚠ The person is diabetic")

        st.warning(
            """
            Recommendations:
            • Maintain healthy diet
            • Exercise regularly
            • Monitor glucose levels
            • Consult healthcare professional
            """
        )

        st.info(f"Confidence Score: {confidence:.2f}%")


st.markdown("---")

st.caption("Built using Machine Learning, Python & Streamlit")