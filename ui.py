import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="AI-Driven Failure Prediction", layout="wide")

# Title and description
st.title("AI-Driven Failure Prediction Model")
st.markdown("""
This application uses machine learning to predict component failures based on historical and live data. 
Upload your dataset, configure the model, and visualize the predictions below!
""")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
else:
    data = None

# Model parameters
st.sidebar.subheader("Model Parameters")
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.5, 0.2, 0.05)
n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100, 10)
random_seed = st.sidebar.number_input("Random Seed", value=42)

# Main content
if data is not None:
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Feature and target selection
    st.subheader("Feature and Target Selection")
    all_columns = data.columns.tolist()
    features = st.multiselect("Select Features", all_columns, default=all_columns[:-1])
    target = st.selectbox("Select Target Variable (Failure: 0 or 1)", all_columns, index=len(all_columns)-1)

    if features and target:
        # Data preparation
        X = data[features]
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Model training
        st.subheader("Model Training")
        with st.spinner("Training the model..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

        # Display results
        st.success("Model training completed!")
        st.write(f"*Accuracy:* {accuracy:.2%}")
        st.write("*Classification Report:*")
        st.write(pd.DataFrame(report).T)

        # Feature importance visualization
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance, ax=ax)
        plt.title("Feature Importance in Failure Prediction")
        st.pyplot(fig)

        # Prediction visualization
        st.subheader("Prediction Distribution")
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig, ax = plt.subplots()
        sns.countplot(x="Predicted", hue="Actual", data=pred_df, ax=ax)
        plt.title("Actual vs Predicted Failures")
        st.pyplot(fig)

        # Live prediction section
        st.subheader("Live Prediction")
        st.write("Enter values for a new component to predict failure:")
        live_input = {}
        for feature in features:
            live_input[feature] = st.number_input(f"{feature}", value=0.0)

        if st.button("Predict"):
            live_data = pd.DataFrame([live_input])
            prediction = model.predict(live_data)[0]
            probability = model.predict_proba(live_data)[0][1]
            st.write(f"*Prediction:* {'Failure' if prediction == 1 else 'No Failure'}")
            st.write(f"*Failure Probability:* {probability:.2%}")

else:
    st.info("Please upload a CSV file to begin. The dataset should contain features and a target column (0 for no failure, 1 for failure).")

# Footer
st.markdown("---")
st.write("Built with EJUST Team")