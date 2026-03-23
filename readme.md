

# 🏥 HealthGuard: High-Sensitivity Clinical Readmission Engine

### **An End-to-End MLOps Framework for Diabetic Risk Stratification**

**Live Deployment Assets:**

  * **Production UI:** [Launch Streamlit Dashboard](https://diabetic-readmission-ml-manish-ba5dqnx4fqxebzp6rfxbnx.streamlit.app/)
  * **Backend:** https://healthguard-api-8cgj.onrender.com/
  * **System Status:** `Operational` | **Target Recall:** `90.0%`

-----

## 1\. Professional System Architecture

Unlike standard monolithic applications, HealthGuard utilizes a **Decoupled Cloud Architecture**. This separation of concerns ensures that the heavy computational load of the Random Forest model does not lag the user interface.

### **The Three-Tier Logic**

1.  **Persistence Layer:** Serialized `.joblib` artifacts containing the trained `Pipeline`, `StandardScaler`, and `LabelEncoder`.
2.  **Logic Layer (FastAPI):** A high-concurrency backend hosted on **Render**. It features Pydantic data validation and an automated **CORS (Cross-Origin Resource Sharing)** policy to allow secure global requests.
3.  **Presentation Layer (Streamlit):** A cloud-native frontend that handles asynchronous API calls, state management, and clinical visualization.

-----

## 2\. Deep-Dive: The Research & Modeling Pipeline

The core of this project is a comparative study of **7 distinct algorithmic configurations**. We didn't just pick a model; we stress-tested the entire Scikit-Learn library.

### **Detailed Performance Matrix**

| Model Architecture | Accuracy | **Recall** | F1-Score | CV Recall (Mean) |
| :--- | :--- | :--- | :--- | :--- |
| **Regularized Random Forest** | **0.6324** | **0.5315** | **0.5741** | **0.5415** |
| Baseline Random Forest | 0.6373 | 0.5780 | 0.5978 | N/A |
| Decision Tree | 0.6242 | 0.5045 | 0.5569 | N/A |
| Support Vector Machine (SVM) | 0.6140 | 0.5105 | 0.5522 | N/A |
| Logistic Regression | 0.6138 | 0.4100 | 0.5000 | N/A |
| k-Nearest Neighbors (kNN) | 0.5853 | 0.4923 | 0.5254 | N/A |
| Naive Bayes | 0.5840 | 0.2300 | 0.3300 | N/A |

### **Why Random Forest Outperformed the Field**

  * **Vs. SVM:** SVM was highly effective but lacked the **Feature Importance** transparency required for clinical trust.
  * **Vs. kNN:** The "Curse of Dimensionality" in our 44-feature space made kNN highly erratic.
  * **Vs. Naive Bayes:** The assumption of feature independence was fatal for clinical data, where "Time in Hospital" and "Number of Procedures" are heavily linked.

-----

## 3\. Engineering Masterclass: Solving the "Clinical Gap"

A model that is 99% accurate but misses the 1% of patients who die is a failure. We solved this through three advanced engineering pillars:

### **Pillar I: SMOTE (Synthetic Minority Over-sampling)**

To combat class imbalance, we didn't just "over-sample" by copying rows. We used **SMOTE** to mathematically interpolate new "Readmitted" data points in the feature space, forcing the model to learn the *boundaries* of risk rather than memorizing the data.

### **Pillar II: Structural Regularization**

To prevent the Random Forest from "memorizing" the training set (Overfitting), we implemented:

  * `max_depth=10`: Limiting tree complexity.
  * `min_samples_leaf=5`: Ensuring every leaf represents a statistically significant group of patients.

### **Pillar III: The 90% Recall Threshold Shift**

Standard models use a 0.5 decision threshold. We recalculated the **Precision-Recall Curve** to find the "Clinical Sweet Spot" at **0.38**. This shift ensures that we capture **90% of all readmissions**, accepting a slightly higher false-alarm rate to guarantee patient safety.

-----

## 4\. Deployment & Infrastructure Setup

This repository is configured for **Environment Parity** (Dev = Prod).

### **Production-Grade Installation**

```powershell
# 1. Virtual Environment Isolation
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate

# 2. Dependency Injection
# We use Scikit-Learn 1.8.0 to match the serialized Joblib version
pip install -r requirements.txt

# 3. Parallel Execution
python app.py & streamlit run frontend.py
```

-----

## 5\. Reflection

This project proves that **MLOps is as important as ML**. By building a decoupled system that handles data cleaning, synthetic sampling, and cross-cloud communication, we have moved from a "Student Experiment" to a "Clinical-Ready Tool." The final system is not just a predictor; it is a scalable infrastructure for hospital decision-making.
