import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st

cv_model = open('predictions.pkl', 'rb')
cv = joblib.load(cv_model)

def prediction(Gender,Age,Salary):
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
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
def predict1(data1):

    vect =cv.texts_to_sequences(data1['v2'])
    vect = pad_sequences(vect, maxlen=max_length, padding=padding_type, truncating=trunc_type)




    predictions=model.predict(vect)

    return predictions
def main ():
    from PIL import Image
    image = Image.open('logo.jpg')
    image_spam = Image.open('images.jpg')
    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict whether to buy iphone or Not')


    st.sidebar.image(image_spam)





    st.title("Iphone Prediction App")

    if add_selectbox == 'Online':
        Age = st.text_input('Age')
        Salary = st.text_input('Salary')
        Gender = st.selectbox('Gender', ["Male","Female"])

        output=""



        if st.button("Predict"):
            result = prediction(Gender, Age,Salary)
            st.success(result)









    if add_selectbox == 'Batch':


        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"],encoding =None, key = 'a')




        st.title('Make sure the csv File is in the same format  as spam.csv before uploading to avoid Error')

        if file_upload is not None:
            data1 = pd.read_csv(file_upload,encoding = 'latin-1')



            predictions =np.asarray(predict1(data1))


            st.write(predictions)



if __name__ == '__main__':
    main()
