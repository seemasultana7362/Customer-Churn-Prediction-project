# 📉 Telco Customer Churn Prediction & Business Impact

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![SHAP](https://img.shields.io/badge/Interpretability-SHAP-purple)

## 📌 Project Overview
Customer retention is critical for telecom and subscription-based companies, as acquiring new customers is significantly more expensive than keeping existing ones. This project presents an end-to-end Machine Learning pipeline that predicts customer churn using historical usage, demographic, and account data from the IBM Telco dataset.

Beyond standard model accuracy, this project focuses on **preventing data leakage**, evaluating models using metrics suited for class imbalance (ROC-AUC & Recall), and utilizing **SHAP (SHapley Additive exPlanations)** to translate machine learning predictions into actionable business strategies.

---

## 🗄️ Dataset & Preprocessing
* **Dataset:** IBM Telco Customer Churn (`telco.csv`)
* **Size:** 7,043 rows, 50 initial columns
* **Target Variable:** `Churn Label` (`Yes` / `No` mapped to `1` / `0`)

### Key Preprocessing Steps
1. **Data Type Correction:** `Total Charges` contained blank string values for new users; these were converted to numeric and missing values filled with `0`.
2. **Data Leakage Mitigation:** Removed features that contain post-churn information or act as direct "cheat codes" for the model (e.g., `Satisfaction Score`, `Churn Score`, `Churn Category`, `Churn Reason`, and `Customer Status`).
3. **Metadata Cleanup:** Dropped non-predictive spatial and ID fields (`Customer ID`, `Zip Code`, `Latitude`, `Longitude`, `Country`, `State`, `City`, `Quarter`).
4. **Encoding & Splitting:** Applied One-Hot Encoding to categorical variables and split the data into an 80/20 train-test ratio with stratification.

---

## ⚙️ Tech Stack & Dependencies
* **Data Handling:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`, `shap`
* **Machine Learning:** `scikit-learn`, `xgboost`

---

## 🚀 Model Benchmarking & Evaluation

Multiple classification models were trained and benchmarked against the test dataset:

| Model | ROC-AUC Score | Primary Focus |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline | Interpretable baseline model |
| **Random Forest** | ~0.89 | Non-linear ensemble decision trees |
| **XGBoost Classifier** | **~0.90+ (Best)** | Gradient boosted trees for optimal decision boundary |

*Note: Models were evaluated primarily on **ROC-AUC** and **Recall** for the churned class to maximize detection of at-risk customers while avoiding false-accuracy inflation.*

---

## 📊 SHAP Interpretability & Business Insights

Using SHAP Tree Explainer on our trained **XGBoost** model, we extracted global feature importances to determine key business drivers:

1. **Contract Type (`Contract_Month-to-month`):** 
   * **Finding:** Customers on Month-to-Month plans exhibit the highest churn probability.
   * **Recommendation:** Offer targeted promotions (e.g., 10–15% discount or free value-adds) to convert Month-to-Month subscribers into 1-Year or 2-Year contracts.
2. **Customer Tenure (`Tenure in Months`):** 
   * **Finding:** Churn risk is concentrated heavily within a customer's first **0 to 12 months**.
   * **Recommendation:** Deploy an automated **90-day onboarding check-in workflow** to resolve early setup issues and improve early retention.
3. **High Monthly Charges & Fiber Optic (`Monthly Charge` / `Internet Type`):** 
   * **Finding:** Fiber Optic subscribers show elevated churn rates, correlated with higher `Monthly Charge` rates.
   * **Recommendation:** Bundle high-value services (such as **Premium Tech Support** or **Device Protection Plan**) into Fiber Optic tiers rather than slashing baseline prices.

---

## 💻 How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/churn-prediction.git](https://github.com/your-username/churn-prediction.git)
   cd churn-prediction
