PROJECT OVERVIEW

Customer churn is one of the biggest challenges in the telecom industry. 

This project uses machine learning techniques to predict which customers are likely to leave based on

behavioral, usage, and account-related features.

THE MODEL PROVIDES:

#Actionable insights for retention campaigns

#Feature importance understanding

#A prediction pipeline that businesses can use in real-time

This project uses the Kangle Telecoms dataset.

OBJECTIVES

#Build and evaluate ML models to predict customer churn

#Identify the most influential factors affecting churn

#Provide business recommendations to reduce customer attrition

#Deploy a clean, reproducible machine learning workflow

Key insights identified during EDA:
#Contract type, monthly charges, and tenure strongly influence churn

#Customers with month-to-month contracts churn more

#High usage + low satisfaction segments are high-risk

#Payment method and support interactions affect churn probability

ML MODELS USED:

#Logistic regression

#Randomforest

Key Steps in the Workflow

1. Data Preprocessing
   
Converted TotalCharges to numeric

Handled missing values using median imputation

Encoded categorical variables

Scaled numerical features

3. Model Selection

A RandomForestClassifier was used because it performs well with mixed datasets and handles non-linear relationships.

3. Model Performance

Classification Report:

Accuracy: 0.80

Churn precision: 0.66

Churn recall: 0.53

Feature Importance (Top Predictors):

TotalCharges

MonthlyCharges

Tenure

Contract Type

Online Security

Payment Method

These insights help telecom businesses focus on customers most likely to churn.

📌 How to Run the Project

1. Clone the Repository

git clone https://github.com/geenova7/customer_churn_project.git
cd customer_churn_project

3. Install Dependencies

pip install -r requirements.txt

5. Run the Script
   
python src/main.py

These insights help telecom businesses focus on customers most likely to churn.
