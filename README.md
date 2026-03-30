# Heart Disease Prediction using Machine Learning 🫀

## 📌 Project Overview
This project is a complete Machine Learning pipeline designed to predict the presence of heart disease in patients based on clinical features. By prioritizing medical diagnostic safety, the model was heavily optimized to minimize False Negatives. The final model achieved an outstanding **ROC-AUC score of 0.94** and successfully identified **90%** of positive disease cases.

## 📊 Dataset
The project utilizes a clinical dataset containing **1,025 patient records** and 14 distinct attributes. Key features include:
* Age & Sex
* Chest Pain Type (Identified as a highly correlated feature)
* Resting Blood Pressure
* Cholesterol Levels
* Maximum Heart Rate Achieved

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning:** Scikit-Learn (SVM, GridSearchCV, RobustScaler)
* **Data Visualization:** Matplotlib, Seaborn
* **Data Manipulation:** Pandas, NumPy

## ⚙️ Machine Learning Pipeline
1. **Exploratory Data Analysis (EDA):** Conducted visual analysis including Target Distribution, Age Distribution, and Top 10 Feature Correlation Heatmaps.
2. **Data Preprocessing:** * Handled categorical variables using One-Hot Encoding (`pd.get_dummies`).
   * Applied `RobustScaler` to standardize features while remaining resilient to clinical outliers (e.g., extreme cholesterol or resting blood pressure values).
3. **Model Selection & Hyperparameter Tuning:** * Algorithm: Support Vector Classifier (`SVC`).
   * Used `GridSearchCV` to test multiple configurations. The optimal parameters found were `{'C': 100, 'kernel': 'linear'}`.
4. **Validation:** Applied 5-fold Cross-Validation to ensure stability, achieving a Mean CV Accuracy of **85.37%**.

## 📈 Evaluation & Key Results
The model's performance on the unseen test set demonstrates high clinical viability:
* **Overall Accuracy:** 84%
* **Recall (Sensitivity for Disease Class):** **90%** (Crucial metric for medical diagnosis)
* **Precision (Disease Class):** 81%
* **ROC-AUC Score:** **0.94** (Indicating excellent separability between healthy and diseased patients)

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Heart-Disease-Prediction-SVM.git](https://github.com/your-username/Heart-Disease-Prediction-SVM.git)
