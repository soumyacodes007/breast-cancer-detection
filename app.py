
import streamlit as st
import pandas as pd
from model_utils import load_model_and_scaler, get_feature_names, make_prediction
from chatbot_module import initialize_chatbot_session, add_message_to_history, display_chat_messages
from gemini_explainer import get_gemini_explanation_with_charts


st.set_page_config(
    page_title="Breast Cancer Predictor & AI Assistant",
    page_icon="🎀",
    layout="wide"
)

#  Load ML Model-
ml_model, scaler = load_model_and_scaler()
feature_names = get_feature_names()


initialize_chatbot_session() # Ensures chatbot and chat_messages exist in session_state


st.title("Breast Cancer Prediction & AI Assistant 🎀")
st.markdown("""
This application uses a Machine Learning model to predict if a breast tumor is benign or malignant.
An AI Assistant is available to help you understand the results and answer questions.
**Disclaimer:** This tool is for educational/informational purposes only and is NOT a substitute for professional medical advice.
""")
st.divider()

# Create two columns: one for inputs/results, one for the chatbot
col_main, col_chatbot = st.columns([2, 1]) # Main content takes 2/3, chatbot 1/3

with col_main:
    st.header("Clinical Feature Input")
    if not feature_names:
        st.error("Could not load feature names. The application cannot proceed.")
    elif not ml_model or not scaler:
        st.error("Machine Learning model or scaler could not be loaded. Prediction unavailable.")
    else:
        with st.form("prediction_form"):
            input_data = {}
            # Dynamically create input fields in 2 columns for better layout
            form_cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with form_cols[i % 2]: # Alternate columns
                    input_data[feature] = st.number_input(
                        label=feature.replace('_', ' ').title(),
                        key=f"input_{feature}",
                        value=st.session_state.get(f"input_val_{feature}", 0.0), # Persist input values
                        step=0.01,
                        format="%.4f"
                    )
            
            submit_button = st.form_submit_button("Predict Tumor Type", type="primary")

        if submit_button:
            # Store current input values in session_state to repopulate form if needed
            for feature, val in input_data.items():
                st.session_state[f"input_val_{feature}"] = val

            prediction, conf_benign, conf_malignant, top_features_list, top_features_dict = make_prediction(
                ml_model, scaler, input_data, feature_names
            )
            
            # Store prediction details in session_state for the chatbot to access if needed
            st.session_state.ml_prediction_details = {
                "prediction_text": prediction,
                "confidence_benign": conf_benign,
                "confidence_malignant": conf_malignant,
                "top_features_list": top_features_list, # List of tuples (name, importance)
                "top_features_dict": top_features_dict, # Dict of {name: importance}
                "input_data": input_data
            }

            st.subheader("🔬 ML Prediction Result:")
            if "Error" not in prediction:
                if prediction == "Malignant":
                    st.error(f"**Prediction: {prediction}**")
                else:
                    st.success(f"**Prediction: {prediction}**")

                res_col1, res_col2 = st.columns(2)
                res_col1.metric(label="Confidence (Benign)", value=f"{conf_benign:.2f}%")
                res_col2.metric(label="Confidence (Malignant)", value=f"{conf_malignant:.2f}%")
                st.caption("Confidence indicates how sure the model is, based on its training data.")

                if top_features_list:
                    st.markdown("---")
                    st.subheader("💡 Top Influential Features (from ML Model)")
                    feature_df = pd.DataFrame(top_features_list, columns=['Feature', 'Importance'])
                    feature_df['Importance'] = feature_df['Importance'].map('{:.4f}'.format)
                    st.table(feature_df.style.hide(axis="index"))
                
                st.markdown("---")
                st.subheader("🤖 AI Explanation of Results (via Gemini)")
                with st.spinner("Generating detailed explanation..."):
                    gemini_explanation_data = get_gemini_explanation_with_charts(
                        prediction, conf_malignant, conf_benign, input_data, top_features_dict
                    )
                
                st.markdown(gemini_explanation_data.get("explanation_text", "No explanation generated."))

                charts_to_render = gemini_explanation_data.get("charts_to_render", [])
                if charts_to_render:
                    st.markdown("---")
                    st.subheader("📊 Visualizations Suggested by AI")
                    for chart_info in charts_to_render:
                        st.markdown(f"**{chart_info.get('title', 'Chart')}**")
                        st.caption(chart_info.get('data_description', ''))
                        if chart_info.get("type") == "bar_chart" and chart_info.get("data_for_chart"):
                            try:
                                # Ensure data_for_chart is a DataFrame or dict suitable for st.bar_chart
                                chart_data = chart_info["data_for_chart"]
                                if isinstance(chart_data, dict):
                                    df_chart = pd.DataFrame(list(chart_data.items()), columns=['Feature', 'Importance'])
                                    st.bar_chart(df_chart.set_index('Feature'))
                                else:
                                    st.bar_chart(chart_data) #  Assumes it's already a plottable structure
                            except Exception as e:
                                st.warning(f"Could not render suggested bar chart: {e}. Data: {chart_info.get('data_for_chart')}")
                        # Add more chart types here if Gemini is prompted to suggest them (e.g., line, scatter)
                        else:
                            st.write(f"Chart type '{chart_info.get('type')}' not yet supported or data missing.")
            else:
                st.error(prediction) # Display error message from make_prediction

            st.markdown("---")
            st.subheader("📋 Your Input Values:")
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df.T.rename(columns={0: 'Value'}))

with col_chatbot:
    st.header("💬 AI Assistant")
    st.caption("Ask follow-up questions about breast cancer, your results, or next steps. (e.g., 'What does 'mean radius' signify?')")

    # Display existing chat messages
    display_chat_messages()

    # Chat input
    user_prompt = st.chat_input("Ask the AI Assistant...")
    if user_prompt:
        add_message_to_history("user", user_prompt)
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chatbot_instance = st.session_state.get('chatbot_instance')
                ml_details_for_chat = st.session_state.get('ml_prediction_details') # Get latest prediction context
                
                if chatbot_instance:
                    response = chatbot_instance.generate_chat_response(user_prompt, ml_details_for_chat)
                    st.markdown(response)
                    add_message_to_history("assistant", response)
                else:
                    error_msg = "Chatbot not initialized. Please refresh."
                    st.error(error_msg)
                    add_message_to_history("assistant", error_msg)
        
        # Optionally, display relevant resources from the chatbot's knowledge base
        if st.session_state.get('ml_prediction_details') and chatbot_instance:
            risk_level = "high_risk" if st.session_state.ml_prediction_details['prediction_text'] == "Malignant" else "low_risk"
            resources = chatbot_instance.get_resources(risk_level)
            if resources:
                with st.expander("📚 Relevant Resources", expanded=False):
                    for resource in resources:
                        st.markdown(f"- {resource}")
st.divider()
st.markdown("---")
st.header("📚 General Educational Content & Resources")
st.markdown("""
- **Early Detection Saves Lives:** Regular screenings and self-examinations are crucial.
- **Know the Symptoms:** Be aware of changes in your breasts and consult a doctor if you notice anything unusual.
- **Useful Links:** [American Cancer Society](https://www.cancer.org/cancer/types/breast-cancer.html), [National Breast Cancer Foundation](https://www.nationalbreastcancer.org/), [BreastCancer.org](https://www.breastcancer.org/)
""")