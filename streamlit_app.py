import streamlit as st
import requests

# Define the URL of the FastAPI service
API_URL = 'http://localhost:8080/predict/'

st.title('Plant Disease Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    files = {'file': uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)
    prediction = response.json()
    st.write('Prediction:', prediction['predicted_class'])
    st.write('Confidence:', prediction.get('confidence', 'N/A'))
  #  st.write('Accuracy:', prediction.get('accuracy', 'N/A'))
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
