import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load face detection model
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Load emotion detection model
model = load_model('model/ResNet50_69_Transfer_Learning.h5')

def detect_and_crop_face(image):
    """Detects a face in the image, draws a rectangle, and crops it."""
    image_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face = image_cv[y:y+h, x:x+w]
        
        # Draw a rectangle around the face
        image_with_rectangle = image_cv.copy()
        cv2.rectangle(image_with_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return Image.fromarray(face), Image.fromarray(image_with_rectangle)
    else:
        return None, None

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

st.markdown(
    "<h3 style='color: #FFD700; text-align: center;'>Please upload an image where your face is clearly visible.</h3>",
    unsafe_allow_html=True
)

st.title("☺ Emotion Detection System")

uploaded_image = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=False)
    
    if st.button("Predict"):
        face_image, image_with_rectangle = detect_and_crop_face(image)  # Detect and crop face
        
        if face_image is None:
            st.markdown("<h3 style='text-align: center; color: red;'>No face detected. Please try another image.</h3>", unsafe_allow_html=True)
        else:
            test_input = preprocess_image(face_image)
            prediction_array = model.predict(test_input)
            
            emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}
            flag = np.argmax(prediction_array)
            predicted_probability = prediction_array[0][flag] * 100
            
            st.image(image_with_rectangle, caption="Detected Face (With Rectangle)", use_container_width=False)
            # st.image(face_image, caption="Cropped Face", use_container_width=False)
            st.markdown(
                f"<h2 style='text-align: center; color: #FFFF00;'>"
                f"Emotion detected: <span style='font-weight: bold; font-size: 48px;'>{emotion_dict[flag]}</span> "
                f"({predicted_probability:.2f}%)</h2>",
                unsafe_allow_html=True,
            )
