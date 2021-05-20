from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('final_decision')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.jpg')
    image_house = Image.open('images.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict whether to buy iphone or Not')


    st.sidebar.image(image_house)





    st.title("Iphone Prediction App")

    if add_selectbox == 'Online':

        Age = st.text_input('Age')
        Salary = st.text_input('Salary')
        Gender = st.selectbox('Gender', ["Male","Female"])



        output=""

        input_dict = {'Age':Age,'Salary':Salary,'Gender':Gender }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)

            if output == 0:
                output='No'
            elif output ==1:
                output='Yes'
        st.success(output)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
