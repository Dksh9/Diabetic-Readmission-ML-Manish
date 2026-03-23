
## Diabetic Readmission Prediction System: End-to-End Clinical Pipeline

### Project Overview

This project has evolved from a comparative study of machine learning algorithms into a complete end-to-end prediction system. The goal is to identify diabetic patients at high risk of hospital readmission within 30 days. By moving beyond simple model training to a full-stack deployment, this system demonstrates how raw clinical data can be transformed into a functional tool for healthcare providers.

The system encompasses data preprocessing, comparative modeling, hyperparameter optimization, model serialization, and a web-based deployment interface using FastAPI and Streamlit.

---

Comparative Analysis: Model Selection Logic
While six different architectures were evaluated, the project moved forward with the Regularized Random Forest for several technical reasons. Below is a summary of why the other models were excluded from the final deployment tier:

1. Naive Bayes: The Baseline Failure
Naive Bayes produced the lowest performance, particularly in Recall (0.23). This indicated that the "independence assumption" of the algorithm does not hold for clinical data. In a hospital setting, features like num_medications and time_in_hospital are often correlated; Naive Bayes failed to capture these dependencies, leading to an unacceptable number of missed high-risk patients.

2. Logistic Regression & Linear SVM: The Linearity Constraint
Both Logistic Regression and Linear Support Vector Machines showed moderate performance but struggled with the high-dimensional, non-linear nature of the dataset. Clinical outcomes are rarely determined by linear relationships between variables. These models were unable to capture the complex interactions between different medications and diagnostic codes that the tree-based models handled with ease.

3. k-Nearest Neighbors (KNN): Scalability and Noise
KNN suffered from the "curse of dimensionality." With 44 features, the distance-based logic became less effective, as the "neighbors" in a high-dimensional space are not always truly similar. Additionally, KNN is sensitive to outliers and noise in clinical lab results, leading to a lack of stability compared to ensemble methods.

4. Decision Tree: The Overfitting Risk
A single Decision Tree was a strong contender but showed a high tendency to overfit the training data. Without the "wisdom of the crowd" found in a Random Forest, the single tree created deep, overly specific rules that did not translate well to the testing set. This resulted in lower overall F1-scores and less reliability for real-world deployment.

Why the Random Forest Won
The Random Forest overcame these individual weaknesses by:

Averaging Errors: By using 200 trees, it cancelled out the errors and noise that plagued the single Decision Tree and KNN.

Handling Non-Linearity: It naturally captured complex patterns that the Linear SVM and Logistic Regression missed.

Clinical Flexibility: It allowed for easier threshold tuning, which was the mechanical key to reaching our 90% Recall target.

---
### Technical Evolution: Before vs. Now

The project underwent a significant overhaul to meet professional clinical standards. Below are the key updates made to the original workflow:

#### 1. From Overfitting to Generalization

Previously, the original Random Forest model achieved high training scores but showed a significant performance gap on unseen data, suggesting it was memorizing noise. We have now implemented a **Regularized Random Forest**. By restricting the maximum depth and increasing the minimum samples per leaf, we narrowed the gap between training and validation scores. This ensures the model generalizes to new patients instead of just the historical dataset.

#### 2. Advanced Imbalance Handling

Initially, we relied primarily on simple class weights. The updated pipeline integrates **SMOTE (Synthetic Minority Over-sampling Technique)** within an imbalanced-learn framework. This generates synthetic examples of the "Readmitted" class, allowing the model to learn the specific characteristics of at-risk patients more effectively.

#### 3. Clinical Threshold Tuning

The model originally used a default 0.50 probability threshold for classification. We have since performed **Threshold Tuning**, shifting the decision cutoff to **0.38**. This increased the **Recall (Sensitivity) to 90%**, ensuring that 9 out of 10 at-risk patients are flagged for review. In healthcare, missing a sick patient (False Negative) is far more dangerous than a false alarm (False Positive).

---

### Performance Leaderboard & Model Selection

The final evaluation determined the **Regularized Random Forest** as the "Champion Model" due to its stability and high sensitivity.

* **Regularized Random Forest (Champion):** 63.24% Accuracy | 0.90 Recall | 0.57 F1-Score.
* **Baseline Random Forest:** 63.74% Accuracy | 0.58 Recall | 0.60 F1-Score (Overfitted).
* **Decision Tree:** 62.43% Accuracy | 0.50 Recall | 0.56 F1-Score.
* **Linear SVM:** 61.40% Accuracy | 0.51 Recall | 0.55 F1-Score.
* **Logistic Regression:** 61.38% Accuracy | 0.41 Recall | 0.50 F1-Score.
* **k-Nearest Neighbors:** 58.54% Accuracy | 0.49 Recall | 0.53 F1-Score.
* **Naive Bayes:** 58.40% Accuracy | 0.23 Recall | 0.33 F1-Score.

---

### System Capabilities

The current system is structured as a three-tier architecture:

* **Research Tier:** Jupyter Notebooks containing the data cleaning, SMOTE application, and 5-fold cross-validation results.
* **API Tier (FastAPI):** A backend server that loads the serialized model (`.joblib`) and provides a REST endpoint for real-time predictions.
* **Interface Tier (Streamlit):** A web dashboard where clinicians can input patient metrics to receive an immediate risk assessment.

---

### Setup and Usage Guide

#### 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn streamlit joblib pandas scikit-learn imbalanced-learn

```

#### 2. Running the System

To launch the full prediction suite, you must run the backend and frontend simultaneously.

**Start the FastAPI Backend:**

```powershell
python app.py

```

The server loads the champion model and waits for data at the local host.

**Start the Streamlit Frontend:**
Open a new terminal, activate the environment, and run:

```powershell
streamlit run frontend.py

```

This opens the browser interface for model interaction.

---

### Final Reflection

This project demonstrates the transition from theoretical data science to applied machine learning. The most critical takeaway was the implementation of **Threshold Tuning**. By intentionally sacrificing some precision to achieve **90% Recall**, the system aligns with clinical priorities: patient safety. The modular structure—separating the model training from the API and the UI—ensures the system is scalable and professional, fitting the requirements for a high-level technical submission.

