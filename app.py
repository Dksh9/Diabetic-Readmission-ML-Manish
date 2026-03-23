from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Diabetic Readmission API")


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
        "recall_goal": 0.90,
        "clinical_threshold": threshold
    }

@app.post("/predict")
def predict_readmission(patient: Patient):
    # 1. Convert to DataFrame
    input_df = pd.DataFrame([patient.data])
    
  
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # 3. Get Probability
    prob = model.predict_proba(input_df)[:, 1][0]
    
   
    if prob >= 0.65:
        risk_level = "🚨 High Risk"
        recommendation = "Immediate clinical intervention required. Review discharge plan and schedule follow-up within 48 hours."
    elif prob >= threshold:
        risk_level = "⚠️ Moderate Risk"
        recommendation = "Flag for care coordination. Ensure medication reconciliation and schedule follow-up within 7 days."
    else:
        risk_level = "✅ Low Risk"
        recommendation = "Proceed with standard discharge protocol and provide educational materials."
    
    return {
        "prediction": risk_level,
        "probability": f"{prob:.2%}",
        "recommendation": recommendation,
        "raw_prob": float(prob) # Helpful for frontend logic
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)