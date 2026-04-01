# ❤️ Heart Disease Risk Predictor

An end-to-end Machine Learning web application that predicts the risk of heart disease based on patient medical data. This project combines **data analysis, model comparison, and deployment using Streamlit** to create an interactive prediction system.

---

## 🚀 Features

* Predicts heart disease risk using Machine Learning
* Displays **probability score** for better interpretation
* Interactive and user-friendly UI built with Streamlit
* Real-time prediction based on medical inputs
* Includes **EDA, model comparison, and model selection**

---

## 📊 Exploratory Data Analysis (EDA)

Performed detailed data analysis using:

* **Seaborn**
* **Matplotlib**

### Key Analysis Performed:

* Distribution of features (Age, Cholesterol, MaxHR, etc.)
* Count plots for categorical variables
* Correlation heatmap to understand relationships
* Feature vs target analysis (important for model selection)

---

## 🧠 Models Evaluated

The following models were trained and compared:

| Model                        | Accuracy    | F1 Score    |
| ---------------------------- | ----------- | ----------- |
| Logistic Regression          | **Highest** | **Highest** |
| K-Nearest Neighbors (KNN)    | High        | High        |
| Support Vector Machine (SVM) | Moderate    | Moderate    |
| Gaussian Naive Bayes         | High        | Moderate    |
| Decision Tree                | Low         | Low         |

---

## 🎯 Model Selection

**Selected Model: Logistic Regression**

### ✅ Why Logistic Regression?

* Achieved the **highest F1-score and accuracy**
* More **stable and reliable** compared to KNN and Naive Bayes
* Provides **better interpretability**, which is crucial in healthcare
* Less prone to overfitting compared to Decision Trees

> Logistic Regression was chosen after comparing multiple models and selecting the one with the best balance of performance and interpretability.

---

## ⚙️ Machine Learning Pipeline

* Data Cleaning & Preprocessing
* Feature Encoding (One-Hot Encoding)
* Feature Scaling (StandardScaler)
* Model Training & Evaluation
* Model Saving using Joblib

---

## 🌐 User Interface (UI)

Built using **Streamlit**, featuring:

* Sidebar-based input system for patient details
* Clean and responsive layout
* Real-time prediction display
* Risk categorization:

  * 🔴 High Risk
  * 🟡 Moderate Risk
  * 🟢 Low Risk
* Probability visualization using progress bar

---

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Seaborn
* Matplotlib
* Joblib

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## 📸 App Preview

(Add your screenshot here)

```
assets/screenshot.png
```

---

## 📌 Future Improvements

* Add model comparison visualization in UI
* Integrate Explainable AI (SHAP)
* Deploy application online
* Improve model accuracy with feature engineering

---

## 💡 Key Learnings

* Importance of **EDA before model training**
* Comparing models using **F1-score instead of only accuracy**
* Building **end-to-end ML pipelines**
* Deploying ML models using Streamlit

---

⭐ If you found this project useful, consider giving it a star!
