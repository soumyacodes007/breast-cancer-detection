
import streamlit as st
import google.generativeai as genai
import json
import pandas as pd

# Using a generally available and capable model.
EXPLAINER_GEMINI_MODEL_NAME = 'gemini-2.0-flash'


def get_gemini_explanation_with_charts(
    ml_prediction: str,
    confidence_malignant: float,
    confidence_benign: float,
    input_features: dict[str, float],
    top_ml_features_dict: dict[str, float] # Dictionary of {feature_name: importance_score}
):
    """
    Prompts Gemini to explain ML results and suggest charts.
    Returns a dictionary with 'explanation_text' and 'charts_to_render'.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return {"explanation_text": "Error: GEMINI_API_KEY not found.", "charts_to_render": []}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(EXPLAINER_GEMINI_MODEL_NAME)

    # Prepare data for the prompt
    input_features_str = json.dumps(input_features, indent=2)
    top_ml_features_str = json.dumps(top_ml_features_dict, indent=2)

    prompt = f"""
    You are an AI assistant helping a user understand their breast cancer prediction results from a machine learning model.
    The results are NOT a medical diagnosis and the user should always consult a doctor.

    ML Model Prediction Details:
    - Prediction: {ml_prediction}
    - Confidence if Malignant: {confidence_malignant:.2f}%
    - Confidence if Benign: {confidence_benign:.2f}%
    - User's Input Clinical Features: {input_features_str}
    - Top 5 Most Influential Features (from the ML model and their importance scores): {top_ml_features_str}

    Task:
    1.  Provide a clear, empathetic, and easy-to-understand explanation of what the ML prediction means (e.g., "The model suggests the characteristics of the tumor are more similar to those typically classified as {ml_prediction}.").
    2.  Explain the confidence score simply (e.g., "The model is {max(confidence_malignant, confidence_benign):.0f}% confident in this prediction based on the data it was trained on.").
    3.  Briefly explain that "influential features" are the inputs that most significantly affected the model's decision for this specific prediction.
    4.  Suggest one or two simple charts to help visualize the results. For each chart, provide:
        a. A "type" (e.g., "bar_chart").
        b. A "title" for the chart.
        c. A "data_description" explaining what the chart shows.
        d. The "data_for_chart" itself, as a JSON object that Streamlit can easily use (e.g., for a bar chart, a dictionary of {{label: value}}).
           - For a bar chart of influential features, use the `top_ml_features_dict` provided.
           - (Optional, if you can infer from inputs): If any input feature seems particularly high or low, you could suggest comparing it to a general understanding, but be very careful not to give medical advice or imply ranges without explicit data for ranges. For now, focus on the influential features bar chart.

    Output Format:
    Return your response as a SINGLE JSON object with two keys: "explanation_text" (string) and "charts_to_render" (list of chart objects).
    Example of a chart object:
    {{
        "type": "bar_chart",
        "title": "Top Influential Features",
        "data_description": "Shows the features that most significantly influenced this prediction and their relative importance.",
        "data_for_chart": {{ "feature_name1": 0.45, "feature_name2": 0.23, ... }}
    }}

    Important Notes:
    -   DO NOT provide medical advice or diagnosis.
    -   Emphasize that this is an AI tool and not a substitute for a doctor.
    -   Keep explanations concise and user-friendly.
    -   Ensure the "data_for_chart" for the bar chart uses the feature names as keys and their importance scores as values from `top_ml_features_dict`.
    """

    try:
        response = model.generate_content(prompt)
        
        # Attempt to parse the JSON response from Gemini
        # Gemini might sometimes add markdown ```json ... ``` around it
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        parsed_response = json.loads(cleaned_response_text)
        
        # Validate structure
        if "explanation_text" not in parsed_response or "charts_to_render" not in parsed_response:
            raise ValueError("Gemini response missing required keys.")
        if not isinstance(parsed_response["charts_to_render"], list):
             raise ValueError("'charts_to_render' should be a list.")

        return parsed_response

    except json.JSONDecodeError as e:
        st.error(f"Error parsing Gemini's explanation JSON: {e}. Raw response: {response.text}")
        return {"explanation_text": "Error: Could not parse explanation from AI. Please check logs.", "charts_to_render": []}
    except Exception as e:
        st.error(f"Error getting explanation from Gemini: {e}")
        return {"explanation_text": "Error: Could not get explanation from AI.", "charts_to_render": []}