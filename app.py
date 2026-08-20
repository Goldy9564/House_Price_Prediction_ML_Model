import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
st.title("Housing Price Prediction using ML")
st.image ("https://www.appliedaicourse.com/blog/wp-content/uploads/2025/01/House-Price-Prediction-Using-Machine-Learning.png")
st.write("Our project, Housing Price Prediction Using Machine Learning, uses data-driven techniques to estimate property prices accurately. By analyzing factors such as location, area, number of rooms, and other features, the model identifies patterns in housing data and predicts prices. This project demonstrates the practical application of Python, data analysis, and machine learning.")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns = housing.feature_names)
df["MedHouseVal"] = housing.target
x = df.iloc[ : , : -2 ]
scaler = MinMaxScaler()
scaled_x= scaler.fit_transform(x)

st.sidebar.title("Select House Features!!🏠🏠")



# Collect input values from sidebar sliders
all_values = []
for column in x.columns:
    value = st.sidebar.slider(f"Select {column} value")
    all_values.append(value)

# Prepare data for prediction
final_data = [all_values]
final_data = scaler.transform(final_data)

# Load the trained model
with open("chatgpt_for_housing.pkl", "rb") as f:
    model = pickle.load(f)
    st.write("✅ Model Loaded Successfully!")

# Make prediction
predicted_price = model.predict(final_data)[0]

# Display result with spinner
with st.spinner("Wait for price prediction..."):
    time.sleep(5)
    st.success(f"Predicted price is: ${np.abs(round(predicted_price * 100000, 2))}")

