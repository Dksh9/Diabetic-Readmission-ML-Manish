# Diabetic Readmission Prediction: A Comparative Machine Learning Study

## Executive Summary

This repository presents a comparative analysis of six machine learning models developed to predict hospital readmission among diabetic patients. In real clinical environments, early identification of high-risk patients is essential for improving outcomes, reducing avoidable readmissions, and lowering healthcare costs.

The objective of this study was not only to compare algorithms, but to understand how different modeling approaches behave when applied to structured clinical data.

---

## Performance Leaderboard

The following results represent the final evaluation on the held-out test set (n = 19,611).

### 1. Random Forest (Top Performer)

* **Accuracy:** 63.74%
* **Recall (Class 1):** 0.58
* **F1-Score:** 0.60
* Best overall balance between sensitivity and specificity for clinical screening purposes.

### 2. Decision Tree

* **Accuracy:** 62.43%
* **Recall (Class 1):** 0.50
* **F1-Score:** 0.56

### 3. Support Vector Machine (Linear)

* **Accuracy:** 61.40%
* **Recall (Class 1):** 0.51
* **F1-Score:** 0.55

### 4. Logistic Regression

* **Accuracy:** 61.38%
* **Recall (Class 1):** 0.41
* **F1-Score:** 0.50

### 5. k-Nearest Neighbors

* **Accuracy:** 58.54%
* **Recall (Class 1):** 0.49
* **F1-Score:** 0.53

### 6. Naive Bayes

* **Accuracy:** 58.40%
* **Recall (Class 1):** 0.23
* **F1-Score:** 0.33

---

## Why Random Forest Performed Best

The Random Forest model outperformed the other approaches primarily due to its ensemble learning strategy. By aggregating the predictions of 200 individual decision trees, it reduced variance and handled noisy clinical features more effectively than single-model approaches.

Several factors contributed to its performance:

* **Class Imbalance Handling:** Using `class_weight='balanced'` helped ensure the minority “Readmitted” class was not overlooked.
* **Non-Linear Feature Relationships:** The model captured complex interactions between variables such as `time_in_hospital`, `num_lab_procedures`, and medication-related features.
* **Robustness to High-Dimensional Data:** Compared to k-Nearest Neighbors and Naive Bayes, the ensemble approach adapted better to sparse and categorical encodings common in healthcare datasets.

---

## Clinical Impact and Conclusion

The primary objective of this project was to maximize recall for the “Readmitted” class. In a hospital setting, missing a high-risk patient is far more costly than generating a false positive.

Improving recall from 0.23 (Naive Bayes) to 0.58 (Random Forest) demonstrates how model selection significantly affects the practical utility of a clinical screening system. While no model achieved near-perfect performance, the Random Forest provided the most reliable trade-off between detecting at-risk patients and maintaining overall model stability.

This study reinforces the idea that healthcare prediction tasks require careful metric selection and architecture choice rather than reliance on a single evaluation measure.

---

## Setup and Usage

1. Clone the repository:
   `git clone [Your Repo Link]`

2. Install dependencies:
   `pip install pandas scikit-learn matplotlib seaborn`

3. Run the summary script:
   `python generate_summary_plots.py`

---

## Self-Reflection and Critical Analysis

This project served as a complete end-to-end exploration of the machine learning pipeline, from preprocessing raw clinical data to evaluating optimized ensemble models. One of the most important lessons for me was understanding how misleading accuracy can be in imbalanced healthcare datasets. If I had relied solely on accuracy, the Naive Bayes model might have appeared acceptable despite failing to identify most high-risk patients.

Another key takeaway was the importance of thoughtful feature engineering. Although the Random Forest delivered the strongest performance, I believe further refinement of diagnostic codes—such as grouping ICD-9 codes by organ system—could reduce dimensionality and potentially improve generalization. The performance gap between linear models and ensemble methods also made it clear that the underlying biological and clinical processes are inherently non-linear, and the model architecture must reflect that complexity.

Overall, this work strengthened my understanding of model evaluation, class imbalance handling, and the trade-offs involved in deploying machine learning systems in real-world clinical contexts.
