# Heart Disease Prediction System 🫀

An end-to-end machine learning web application and classification pipeline designed to predict the likelihood and severity of cardiovascular disease based on clinical parameters. Built with Python, Streamlit, Scikit-learn, XGBoost, and PostgreSQL.

---

## 🚀 Features

* **Interactive Web Interface:** User-friendly UI built with Streamlit for seamless data input and real-time risk assessment.
* **Advanced Imputation Pipeline:** Handles missing clinical data using predictive modeling (`RandomForestRegressor`/`Classifier` via `IterativeImputer`) instead of basic statistical fill methods.
* **Robust ML & Tuning Pipeline:** Evaluates and optimizes multiple classification algorithms using cross-validation and hyperparameter tuning (`GridSearchCV`).
* **Database Integration:** Securely connects to a PostgreSQL database to store patient inputs, prediction history, and system logs.
* **Data Visualization & EDA:** Exploratory data analysis, correlation distributions, and feature analysis by demographics.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Plotly, Matplotlib, Seaborn
* **Machine Learning & Data:** Python, Scikit-learn, XGBoost, Pandas, NumPy, Imbalanced-learn
* **Database:** PostgreSQL
* **Version Control:** Git & GitHub

---

## 🔬 Methodological Approaches & Strategies

To build a robust and reliable cardiovascular prediction system, several machine learning strategies and iterative improvements were tested and implemented:

### 1. Advanced Missing Data Imputation
Instead of relying on basic statistical imputation (mean/median), the pipeline utilizes predictive modeling for missing clinical values:
* **Categorical & Binary Features:** Imputed using `RandomForestClassifier` models trained on the remaining features.
* **Continuous Features:** Imputed using `RandomForestRegressor` and `IterativeImputer` to preserve realistic data distributions and correlations.

### 2. Feature Engineering & Binning
* Discretized continuous variables (such as resting blood pressure - `trestbps`) into clinically relevant risk thresholds (hypotension, normal, prehypertension, stage 1 hypertension, severe hypertension) to improve model interpretability.
* Cleaned anomalous records (e.g., physiological zeroes in blood pressure or cholesterol).

### 3. Data Transformation & Scaling
* Applied `StandardScaler` and `PowerTransformer` (Yeo-Johnson method) to normalize continuous features, stabilize variance, and mitigate the impact of outliers—optimizing performance particularly for distance-based models.

### 4. Algorithm Exploration, Comparison & Tuning
To find the model with the best predictive performance, we tested and compared a wide array of classification algorithms via cross-validation (5-fold CV):
* **Linear & Probabilistic Models:** Logistic Regression, Naive Bayes.
* **Distance & Neighborhood-Based Models:** K-Nearest Neighbors (KNN), Support Vector Classification (SVC).
* **Tree-Based & Ensemble Methods:** Decision Trees, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.
* **Optimization & Evaluation:** Experimented with both multi-class target configurations (`num` 0–4) and binary/reduced multi-class mappings to evaluate clinical efficacy and maximize metrics like Weighted Accuracy, Macro F1-score, and ROC-AUC. Hyperparameters were optimized using `GridSearchCV` combined with feature selection pipelines (`SelectKBest`).


---



## 📁 Project Structure

```text
heart-disease-prediction/
│
├── venv/                     # Python virtual environment
├── __pycache__/              # Python compilation cache
├── .env                      # Environment variables configuration
├── app                       # Main Streamlit application script
├── database                  # PostgreSQL connection & query management
├── model.ipynb               # Jupyter/Python source script for training & evaluation
├── encoders.pkl              # Saved encoders for categorical features
├── feature_names.pkl         # Saved feature columns list
├── model_random_forest.pkl   # Saved binary for the best performing model (Random Forest)
├── scaler.pkl                # StandardScaler object binary
├── yeojohnson.pkl            # PowerTransformer object binary
└── README.md                 # Project documentation

