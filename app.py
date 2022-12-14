
import streamlit as st
import pandas as pd
import csv
import altair as alt
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#1
s = pd.read_csv("social_media_usage.csv")
print(s.shape) #dimension

#2
def clean_sm(x):
    value = np.where(x.iloc[:,[1]]==1,1,0)
    x["value"] = value
    print(x)
    

#3
#New dataframe and data cleaning
ss = s.loc[:,["web1h","income", "educ2", "par", "marital", "gender", "age"]]
ss.rename(columns={"web1h":"sm_li", "educ2": "education"},inplace=True)
#Assigning variable features, dropping missing values, assigning data types
ss["sm_li"] = np.where(ss["sm_li"]==1,1,0)
ss["education"] = np.where(ss["education"] <=8, ss["education"], np.nan)
ss["income"] = np.where(ss["income"] <=9, ss["income"], np.nan)
ss["age"] = np.where(ss["age"] <99, ss["income"], np.nan)
ss["par"] = np.where(ss["par"]==1,1,0)
ss["marital"] = np.where(ss["marital"]==1,1,0)
ss["gender"] = np.where(ss["gender"]==1,1,0)
ss = ss.dropna()
ss = ss.astype({"education": int, "income": int, "age":int})



#4: Target(y) and feature (x) selection
y = ss["sm_li"]
X = ss[["education", "income", "age", "par", "marital", "gender"]]

#5: Data Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=750)

#6: Logistic Regression Model
lr = LogisticRegression(random_state=750, class_weight="balanced")

lr_balanced = lr.fit(X_train, y_train)


#App Header
st.markdown("# Predicting LinkedIn Users Application")

st.markdown("#### The objective of this application is to utilize a Machine Learning Logistic Regression Model to predict whether or not an individual is a LinkedIn user based on the responses below.")

#"User Input"
income = st.selectbox(label="What is your income level?",
options=("<$10k", "$10k - $20k", "$20k - $30k", "$30k - $40k","$40k - $50k", "$50k - $75k", "$75k - $100k", "$100k - $150k", "$150k+" ))

if income == "<$10k":
    income = 1
elif income == "$10k - $20k":
    income = 2
elif income == "$20k - $30k":
    income = 3
elif income == "$30k - $40k":
    income = 4
elif income == "$40k - $50k":
    income = 5
elif income == "$50k - $75k":
    income = 6
elif income == "$75k - $100k":
    income = 7
elif income == "$100k - $150k":
    income = 8
elif income == "$150k+":
    income = 9


education = st.selectbox(label="What is your highest level of education?",
options=("Less than High School", "High School incomplete", "High School graduate", "Some college, no degree", "Associate Degree", "Bachelor's Degree", "Some Postgraduate school", "Postgraduate Degree"))

if education == "Less than High School":
    education = 1
elif education == "High School incomplete":
    education = 2
elif education == "High School graduate":
    education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Associate Degree":
    education = 5
elif education == "Bachelor's Degree":
    education = 6
elif education == "Some Postgraduate school":
    education = 7
elif education == "Postgraduate Degree":
    education = 8

par = st.selectbox(label="Are you a parent?",
options=("Yes", "No"))

if par == "Yes":
    par = 1
elif par == "No":
    par = 2

marital = st.selectbox(label="What is your current marital status?",
options=("Married", "Living with a partner", "Divorced", "Separated", "Widowed", "Never been Married"))

if marital == "Married":
    marital = 1
elif marital == "Living with a partner":
    marital = 2
elif marital == "Divorced":
    marital = 3
elif marital == "Separated":
    marital = 4
elif marital == "Widowed":
    marital = 5
elif marital == "Never been Married":
    marital = 6

gender = st.selectbox(label="What is your gender?",
options=("Male", "Female", "Other"))
if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 2
elif gender == "Other":
    gender = 3

age = st.slider("What is your age?", 1, 99)


education = int(education)
income = int(income)
par = int(par)
marital = int(marital)
gender = int(gender)



prediction = lr_balanced.predict([[education, income, age, par, marital, gender]])

if prediction == 0:
    st.success("Not a LinkedIn User")
elif prediction == 1:
    st.success("LinkedIn User")
