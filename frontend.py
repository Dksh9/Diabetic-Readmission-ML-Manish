import streamlit as st
import requests

st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")

st.title("🏥 Patient Readmission Risk Tool")
st.markdown("""
This tool uses a **Regularized Random Forest** (Recall: 90%) to identify at-risk diabetic patients.
It evaluates clinical metrics to provide risk stratification and actionable next steps.
""")

st.sidebar.header("Patient Metrics")


time_in_hospital = st.sidebar.slider("Time in Hospital (Days)", 1, 14, 3)
num_lab_procedures = st.sidebar.number_input("Number of Lab Procedures", 1, 150, 40)
num_medications = st.sidebar.number_input("Number of Medications", 1, 100, 15)

# When user clicks predict
if st.button("Analyze Risk", use_container_width=True):
    
    payload = {
        "data": {
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications
        }
    }
    
  
    try:
        response = requests.post("https://healthguard-api-8cgj.onrender.com/predict", json=payload)
        result = response.json()
        
        st.divider()
        
    
        if "High" in result['prediction']:
            st.error(f"**Result: {result['prediction']}**")
            st.metric("Readmission Probability", result['probability'])
        elif "Moderate" in result['prediction']:
            st.warning(f"**Result: {result['prediction']}**")
            st.metric("Readmission Probability", result['probability'])
        else:
            st.success(f"**Result: {result['prediction']}**")
            st.metric("Readmission Probability", result['probability'])
        
        
        st.subheader("📋 Recommended Clinical Actions")
        st.info(result['recommendation'])
            
    except Exception as e:
        st.error("Make sure the FastAPI server (app.py) is running!")