import streamlit as st
import tensorflow as tf
from PIL import Image
import time
import numpy as np

# Page config
st.set_page_config(page_title="Plant Disease Detection App",
                   page_icon="images/logo-01.png")

# Page title
st.title("Plant Disease Detection")
st.image("images/logo-02.png")
st.write("\n\n")

# Load the TFLite model and labels
interpreter = tf.lite.Interpreter(model_path="model/plant_model.tflite")
interpreter.allocate_tensors()
class_names = ['Healthy', 'Powdery', 'Rust']

preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    # image = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    image = image.resize((224, 224))

    # Convert to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Normalize the image
    # image = image / 255.0
    image = preprocess_input(image)

    return image


# Function to make predictions
def predict(image, class_names):
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_tensor_index, preprocessed_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predicted_class = interpreter.get_tensor(output_tensor_index)

    # Get the predicted class
    predicted_class_name = class_names[int(predicted_class.argmax())]

    probability = [predicted_class] * 100

    return predicted_class_name, probability


# Streamlit app
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Make prediction when button is clicked
    if st.button("Classify"):
        start_time = time.time()
        predicted_class_name, probability = predict(image, class_names)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        st.success(f"Predicted Class: {predicted_class_name} with Confidence {probability[0]:.2f}"
                   f" in {inference_time:.2f} ms")
