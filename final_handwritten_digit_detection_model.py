

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Title
st.markdown("<h1 style='text-align: center; color: dark blue; font-size: 60px;'>Handwritten Digit Recognition Application</h1>", unsafe_allow_html=True)

# Header
st.markdown("<h2 style='text-align: center;'><u>Based on Artificial Neural Network</u></h2>", unsafe_allow_html=True)

# Subheader
st.subheader("Dataset")

# Text
st.text("I have trained the model on MNIST dataset having 70k images")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # x_train = 60,000; test set = 10,000

# Display shape of training data
st.write("Shape of training data:", x_train.shape)
st.write("Shape of test data:", x_test.shape)

# Standardize the data to make convergence faster
X_train = x_train / 255.0  # Centralize data between 0 and 1
X_test = x_test / 255.0

# Load the model
model_path = "C:\\Users\\Babar Raheem\\handwritten digit classification model//handwritten_digit_detection.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Predict probabilities on test data
y_prob = loaded_model.predict(X_test)  # Predict the probability of each input digit (0-9)

# Find the digit with the highest probability
y_predict = y_prob.argmax(axis=1)

# Check the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predict)
st.subheader("ACCURACY")
st.write("Accuracy of the model:", accuracy)

# Define function to visualize prediction
def visualize_prediction(index):
    fig, ax = plt.subplots()
    ax.imshow(X_test[index], cmap='gray')
    st.pyplot(fig)

    # Display the model's prediction for this example
    predicted_label = loaded_model.predict(X_test[index].reshape(1, 28, 28)).argmax(axis=1)
    st.write("Model's prediction for the displayed digit:", predicted_label[0])

def main():
    st.title('Model Prediction Visualizer')
    st.write("Enter the index of the example you want to visualize:")
    index = st.number_input("Index", min_value=0, max_value=len(X_test)-1, step=1, value=50)
    
    visualize_prediction(index)

if __name__ == "__main__":
    main()
