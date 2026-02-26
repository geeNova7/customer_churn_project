import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#load data
df = pd.read_csv('../data/telco_customer_churn.csv')
print(df.head())
print(df.info())

#data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median())

#encorder for categorial feature
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']):
    df[col] = le.fit_transform(df[col])

#split data
x = df.drop('Churn',axis=1)
y = df['Churn']

train_x,test_x,train_y,test_y = train_test_split(x,y ,test_size=0.2,random_state=33)

#train model
churn_model = RandomForestClassifier()
churn_model.fit(train_x,train_y)

#evaluate model
churn_pred = churn_model.predict(test_x)

print("confusion matrix:",confusion_matrix(test_y,churn_pred))
print("\nclassification report:\n",classification_report(test_y,churn_pred))

#save evaluation
with open('../output/churn_report.txt','w') as f:
    f.write("classifiaction report:\n")
    f.write(classification_report(test_y,churn_pred))

#feature importance
importance = churn_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature':x.columns,
    'importance':importance
}).sort_values(by='importance',ascending=False)

print("\nfeature importance:",feature_importance)

#save model
joblib.dump(churn_model,'../output/churn_model.pkl')

print("\nmodel saved to ../output/churn_model.pkl")