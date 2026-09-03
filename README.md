# 📊 Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to **churn or stay** based on customer demographics, services, contract details, and billing information.

## 🚀 Project Overview

Customer churn is an important problem for telecom companies. Predicting customers who are likely to leave can help businesses take preventive actions and improve customer retention.

This project covers the complete machine learning workflow:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature transformation
* Model training
* Model evaluation
* Hyperparameter tuning
* Model and preprocessor saving
* Streamlit deployment

## 🧠 Machine Learning Approach

Several classification models were evaluated during the project, including:

* Logistic Regression
* Decision Tree
* Random Forest
* Balanced Logistic Regression
* Tuned Random Forest

The Random Forest model was further tuned using cross-validation to find better hyperparameters.

### Best Random Forest Parameters

```text
n_estimators = 100
min_samples_split = 5
min_samples_leaf = 2
max_features = log2
max_depth = 8
```

### Final Tuned Random Forest Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 75.23% |
| Precision | 52.26% |
| Recall    | 77.27% |
| F1 Score  | 62.35% |

Recall and F1-score were given particular attention because correctly identifying customers who may churn is important for this problem.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_random_forest.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirement.txt
```

## 🌐 Streamlit Application

The trained model and preprocessing pipeline are integrated into a Streamlit application.

The application allows users to enter customer information such as:

* Gender
* Senior citizen status
* Partner and dependents
* Tenure
* Phone and internet services
* Online security and backup
* Device protection
* Tech support
* Streaming services
* Contract type
* Payment method
* Monthly charges
* Total charges

The application then predicts the customer's churn risk and displays:

* Stay probability
* Churn probability
* Churn risk

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 📌 Future Improvements

* Improve model performance through additional feature engineering
* Compare additional ensemble models
* Add explainable AI features
* Improve the Streamlit dashboard
* Deploy the application publicly

## 👩‍💻 Author

**Khushbu Kumari**

B.Tech CSE (AI & ML) Student

Interested in Machine Learning, Data Science and Software Development.
