# model_utils.py
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer # To get feature names easily
import streamlit as st # For caching

# --- Constants ---
MODEL_PATH = 'breast_cancer_rfc_model.joblib'
SCALER_PATH = 'breast_cancer_scaler.joblib'

@st.cache_resource # Cache the loading of model and scaler
def load_model_and_scaler():
    """Loads the trained model and scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully from model_utils.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the root directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

@st.cache_data # Cache the feature names as they don't change
def get_feature_names():
    """Gets the feature names from the breast cancer dataset."""
    try:
        cancer_data = load_breast_cancer()
        return list(cancer_data.feature_names)
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        # Fallback or define manually if needed, but this is less robust
        return [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

def make_prediction(model, scaler, input_data, feature_names_ordered):
    """
    Makes a prediction using the loaded model and scaler.
    Args:
        model: The trained machine learning model.
        scaler: The fitted scaler.
        input_data (dict): A dictionary where keys are feature names and values are user inputs.
        feature_names_ordered (list): The list of feature names in the order the model expects.
    Returns:
        tuple: (prediction_text, confidence_benign, confidence_malignant, top_features)
    """
    if model is None or scaler is None:
        return "Error: Model not loaded.", 0, 0, []

    try:
        # Ensure input_data is in the correct order
        ordered_input_values = [float(input_data[feature]) for feature in feature_names_ordered]
        final_features = np.array([ordered_input_values])

        scaled_features = scaler.transform(final_features)
        prediction_val = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)

        result_text = "Benign" if prediction_val[0] == 1 else "Malignant"
        confidence_benign = probabilities[0][1] * 100
        confidence_malignant = probabilities[0][0] * 100

        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_dict = dict(zip(feature_names_ordered, importances))
            sorted_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
            top_features = sorted_importances[:5] # Top 5 features

        return result_text, confidence_benign, confidence_malignant, top_features

    except ValueError as ve:
        st.error(f"Invalid input: Please ensure all fields are numbers. ({ve})")
        return "Error in input.", 0, 0, []
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Prediction error.", 0, 0, []