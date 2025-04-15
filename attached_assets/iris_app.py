import streamlit as st
import pandas as pd
import numpy as np
from iris import models, scaler, predict_species  # Import from iris.py

# Streamlit app title
st.title("Iris Species Predictor")

# Sidebar for user input
st.sidebar.header("Input Flower Specifications")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Select a model
model_name = st.sidebar.selectbox(
    "Select a Model",
    ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", 
     "Decision Tree", "Random Forest", "Naive Bayes"]
)

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Prepare input data
        input_data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }

        # Get the selected model
        selected_model = models[model_name]

        # Predict the species
        predicted_species = predict_species(selected_model, input_data)

        # Display the result
        st.success(f"The predicted species is: **{predicted_species}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")