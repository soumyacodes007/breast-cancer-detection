
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import streamlit as st

MODEL_PATH = 'breast_cancer_rfc_model.joblib'
SCALER_PATH = 'breast_cancer_scaler.joblib'

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the root directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

@st.cache_data
def get_feature_names():
    try:
        cancer_data = load_breast_cancer()
        return list(cancer_data.feature_names)
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return [ # Fallback
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

def make_prediction(model, scaler, input_data_dict, feature_names_ordered):
    # Default error return values (5 items)
    error_prediction_text = "Prediction Error"
    error_conf_benign = 0.0
    error_conf_malignant = 0.0
    error_top_features_list = []
    error_feature_importance_dict = {}

    if model is None or scaler is None:
        st.error("Model or scaler not loaded for prediction.")
        return "Error: Model not loaded.", error_conf_benign, error_conf_malignant, error_top_features_list, error_feature_importance_dict

    try:
        # Ensure all input features are present and are convertible to float
        ordered_input_values = []
        for feature in feature_names_ordered:
            if feature not in input_data_dict:
                st.error(f"Missing input value for feature: {feature}")
                return f"Error: Missing input for {feature}.", error_conf_benign, error_conf_malignant, error_top_features_list, error_feature_importance_dict
            try:
                ordered_input_values.append(float(input_data_dict[feature]))
            except ValueError:
                st.error(f"Invalid non-numeric input for feature: {feature}. Value: {input_data_dict[feature]}")
                return f"Error: Invalid input for {feature}.", error_conf_benign, error_conf_malignant, error_top_features_list, error_feature_importance_dict
        
        final_features = np.array([ordered_input_values])
        scaled_features = scaler.transform(final_features)
        
        prediction_val = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)

        result_text = "Benign" if prediction_val[0] == 1 else "Malignant"
        confidence_benign = probabilities[0][1] * 100
        confidence_malignant = probabilities[0][0] * 100

        top_features_list_calc = []
        feature_importance_dict_calc = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_dict_calc = dict(zip(feature_names_ordered, importances))
            sorted_importances = sorted(feature_importance_dict_calc.items(), key=lambda item: item[1], reverse=True)
            top_features_list_calc = sorted_importances[:5]

        return result_text, confidence_benign, confidence_malignant, top_features_list_calc, feature_importance_dict_calc

    except ValueError as ve: # This specifically catches float conversion errors if not caught above
        st.error(f"Invalid input during processing: {ve}")
        return "Error: Invalid input type.", error_conf_benign, error_conf_malignant, error_top_features_list, error_feature_importance_dict
    except Exception as e:
        st.error(f"An unexpected error occurred during ML prediction: {e}")
        return "ML Prediction error (unexpected).", error_conf_benign, error_conf_malignant, error_top_features_list, error_feature_importance_dict