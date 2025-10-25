import streamlit as st
import pandas as pd
from groq import Groq
from PIL import Image

# Initialize Groq client (ensure you set your GROQ_API_KEY as an environment variable)
client = Groq(api_key=st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else None)

def groq_prediction(Gender, Age, Salary):
    """
    Use a Groq large language model to predict if a person will buy an iPhone.
    """
    prompt = f"""
    You are an expert marketing data analyst.
    Predict whether a person will buy an iPhone based on the following details:
    - Gender: {Gender}
    - Age: {Age}
    - Salary: ${Salary}

    Respond only with one word: 'Yes' or 'No'.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # You can change to "llama3-70b-8192" if preferred
        messages=[{"role": "user", "content": prompt}]
    )

    prediction = response.choices[0].message.content.strip()
    return prediction


def main():
    # App header
    image = Image.open('logo.jpg')
    image_spam = Image.open('images.jpg')
    st.image(image,  use_container_width=False)

    st.sidebar.info('This app predicts whether a person will buy an iPhone.')
    st.sidebar.image(image_spam)

    st.title("📱 iPhone Purchase Prediction App (Powered by Groq LLM)")

    # Sidebar selection
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch")
    )

    # Online Prediction
    if add_selectbox == 'Online':
        Age = st.number_input('Age', min_value=10, max_value=100, step=1)
        Salary = st.number_input('Salary (USD/Annum)', min_value=0)
        Gender = st.selectbox('Gender', ["Male", "Female"])

        if st.button("Predict"):
            with st.spinner("🔍 Asking Groq model for prediction..."):
                result = groq_prediction(Gender, Age, Salary)
            st.success(f"Prediction: {result}")

    # Batch Prediction
    elif add_selectbox == 'Batch':
        file_upload = st.file_uploader("📂 Upload CSV file for predictions", type="csv")
        st.info("Ensure the CSV file has columns: Gender, Age, Salary")

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            results = []
            with st.spinner("Generating predictions with Groq..."):
                for _, row in data.iterrows():
                    pred = groq_prediction(row['Gender'], row['Age'], row['Salary'])
                    results.append(pred)

            data['Predicted_Purchase'] = results
            st.success("✅ Predictions complete!")
            st.dataframe(data)

            # Allow user to download results
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='iphone_predictions.csv',
                mime='text/csv'
            )


if __name__ == '__main__':
    main()

