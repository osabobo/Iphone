import numpy as np 
import pandas as pd 
import joblib 
import os 
import streamlit as st 
from sklearn.preprocessing import LabelEncoder 
cv_model = open('predictions.pkl', 'rb') 
cv = joblib.load(cv_model)
def prediction(Gender,Age,Salary): if Gender == "Male": Gender = 0 else: Gender = 1
Age=Age

Salary=Salary
 # Making predictions

prediction = cv.predict(
    [[Gender,Age,Salary]])
if prediction == 0:
    pred = 'No'
else:
    pred = 'Yes'
return pred
def main (): from PIL import Image image = Image.open('logo.jpg') image_spam = Image.open('images.jpg') st.image(image,use_column_width=False)
add_selectbox = st.sidebar.selectbox(
"How would you like to predict?",
("Online", "Batch"))

st.sidebar.info('This app is created to predict whether to buy iphone or Not')


st.sidebar.image(image_spam)





st.title("Iphone Prediction App")

if add_selectbox == 'Online':
    Age = st.text_input('Age')
    Salary = st.text_input('Salary(USD/Annum)')
    Gender = st.selectbox('Gender', ["Male","Female"])

    result=""




    if st.button("Predict"):
        result = prediction(Gender, Age,Salary)
        st.success(result)
