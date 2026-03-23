import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Diabetic Readmission API")

# --- FIX 1: Allow Streamlit to talk to this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd put your streamlit URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
artifacts = joblib.load('champion_rf_readmission_model.joblib')
model = artifacts['pipeline']
threshold = artifacts['optimal_threshold']
feature_names = artifacts['feature_names']

class Patient(BaseModel):
    data: dict

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model": "Regularized Random Forest", 
        "clinical_threshold": threshold
    }

@app.post("/predict")
def predict_readmission(patient: Patient):
    input_df = pd.DataFrame([patient.data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    prob = model.predict_proba(input_df)[:, 1][0]
    
    if prob >= 0.65:
        risk_level = "🚨 High Risk"
        recommendation = "Immediate clinical intervention required."
    elif prob >= threshold:
        risk_level = "⚠️ Moderate Risk"
        recommendation = "Flag for care coordination."
    else:
        risk_level = "✅ Low Risk"
        recommendation = "Standard discharge protocol."
    
    return {
        "prediction": risk_level,
        "probability": f"{prob:.2%}",
        "recommendation": recommendation,
        "raw_prob": float(prob)
    }

if __name__ == "__main__":
    # --- FIX 2: Dynamic Port for Cloud Deployment ---
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)