import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load the models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
rnn_model = tf.keras.models.load_model('rnn_model.h5')

# Class names and mapping
class_names = ['ethiopia', 'malawi', 'nigeria']  # Update with actual class names if necessary
poverty_mapping = {
    "ethiopia": "High Poverty",
    "malawi": "Medium Poverty",
    "nigeria": "Low Poverty"
}

# Image upload section
st.title("Poverty Level Classification")
st.write("Upload an image to predict the poverty level of the region.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image with a reduced size
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", width=300)  # Resize to 300px width
    st.write("")

    # Function to predict the image and display the result
    def predict_image(model, img):
        img_array = image.img_to_array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = model.predict(img_array, verbose=0)

        predicted_class = np.argmax(predictions)
        class_name = class_names[predicted_class]
        poverty_level = poverty_mapping[class_name]

        return class_name, poverty_level, predictions

    # Prediction using CNN
    cnn_class, cnn_poverty_level, cnn_preds = predict_image(cnn_model, img)

    # Display CNN Prediction
    st.subheader(f"CNN Prediction: {cnn_class}")
    st.write(f"Poverty Level: {cnn_poverty_level}")
    st.write(f"Prediction Probabilities: {cnn_preds}")

    # Prediction using RNN
    rnn_class, rnn_poverty_level, rnn_preds = predict_image(rnn_model, img)

    # Display RNN Prediction
    st.subheader(f"RNN Prediction: {rnn_class}")
    st.write(f"Poverty Level: {rnn_poverty_level}")
    st.write(f"Prediction Probabilities: {rnn_preds}")

    # Display the prediction result
    st.write("### Prediction Result")
    st.write(f"CNN Model Prediction: {cnn_class} ({cnn_poverty_level})")
    st.write(f"RNN Model Prediction: {rnn_class} ({rnn_poverty_level})")

    # Plotting CNN and RNN Prediction Probabilities
    fig, ax = plt.subplots()
    ax.bar(class_names, cnn_preds[0])
    ax.set_title('CNN Prediction Probabilities')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.bar(class_names, rnn_preds[0])
    ax.set_title('RNN Prediction Probabilities')
    st.pyplot(fig)
