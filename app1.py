import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import sklearn
from PIL import Image

# --- Safe model loading with version compatibility check ---
MODEL_PATH = 'predictions.pkl'

def load_model(model_path=MODEL_PATH):
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        st.sidebar.success(f"✅ Model loaded successfully (scikit-learn {sklearn.__version__})")
        return model
    except Exception as e:
        st.sidebar.error("❌ Failed to load model.")
        st.sidebar.warning(
            f"Possible version mismatch:\n\n"
            f"- Installed scikit-learn: {sklearn.__version__}\n"
            f"- Error: {str(e)}\n\n"
            f"💡 Try reinstalling the same scikit-learn version used during training "
            f"or retrain and resave your model using this version."
        )
        st.stop()

cv = load_model()

# --- Prediction function ---
def prediction(Gender, Age, Salary):
    try:
        Gender = 0 if Gender == "Male" else 1
        Age = float(Age)
        Salary = float(Salary)
        pred = cv.predict([[Gender, Age, Salary]])
        return 'Yes' if pred[0] == 1 else 'No'
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Streamlit main app ---
def main():
    image = Image.open('logo.jpg')
    image_spam = Image.open('images.jpg')

    st.image(image, use_column_width=False)
    st.sidebar.image(image_spam)
    st.sidebar.info('This app predicts whether a person will buy an iPhone or not.')

    mode = st.sidebar.selectbox("Choose Prediction Mode", ("Online", "Batch"))

    st.title("📱 iPhone Purchase Prediction App")

    if mode == 'Online':
        Gender = st.selectbox('Gender', ["Male", "Female"])
        Age = st.text_input('Age')
        Salary = st.text_input('Salary (USD/Annum)')

        if st.button("Predict"):
            result = prediction(Gender, Age, Salary)
            if result:
                st.success(f"Prediction: {result}")

    elif mode == 'Batch':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")

        st.info('Ensure your CSV matches the format of `iphone.csv` before uploading.')

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
                if 'Purchase Iphone' in data.columns:
                    data = data.drop('Purchase Iphone', axis=1)

                predictions = cv.predict(data)
                data['Prediction'] = np.where(predictions == 1, 'Yes', 'No')
                st.write(data)
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

if __name__ == '__main__':
    main()
