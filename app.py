import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸"
)

# Title
st.title("🌸 Iris Flower Prediction")
st.write("Predict Iris flower species using Machine Learning.")

# Load dataset
data = pd.read_csv("Iris.csv")

# Show dataset
st.subheader("Iris Dataset")
st.dataframe(data.head())

# Features and target
X = data[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]

y = data["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("### Model Accuracy")
st.success(f"{accuracy * 100:.2f}%")

# User input
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# Prediction
if st.button("Predict 🌸"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Species: {prediction[0]}")