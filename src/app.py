# install tensorflow of 2.15
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    test_input = np.expand_dims(image_array, axis=0)
    return test_input



model = load_model('model\ResNet50_69_Transfer_Learning.h5')
st.title("☺ Emotion Detection System")

uploaded_image = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=False)


    if st.button("Predict"):
        test_input = preprocess_image(image)
        prediction_array = model.predict(test_input)
        emotion_dict = {
            0: "Angry",
            1: "Happy",
            2: "Neutral",
            3: "Sad",
            4: "Surprise"
        }
        flag = np.argmax(prediction_array)
        predicted_probability = prediction_array[0][flag] * 100
        st.markdown(
                f"<h2 style='text-align: center; color: #FFFF00;'>"
                f"Emotion detected: <span style='font-weight: bold; font-size: 48px;'>{emotion_dict[flag]}</span> "
                f"({predicted_probability:.2f}%)</h2>",
                unsafe_allow_html=True,
            )
