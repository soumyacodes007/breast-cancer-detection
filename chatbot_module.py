# chatbot_module.py
import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai

# --- Constants ---
# Using a generally available and capable model.
# Options: 'gemini-1.0-pro', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

class BreastCancerChatbot:
    def __init__(self):
        self.model = None
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY not found chck env file.")
                return
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        except Exception as e:
            st.error(f"Error initializing Gemini model: {e}")
            self.model = None # Ensure model is None if initialization fails

        self.knowledge_base = {
            "high_risk": {
                "resources": [
                    "American Cancer Society: https://www.cancer.org/cancer/types/breast-cancer.html",
                    "National Breast Cancer Foundation: https://www.nationalbreastcancer.org/"
                ]
            },
            "low_risk": {
                "resources": [
                    "BreastCancer.org: https://www.breastcancer.org/",
                    "CDC Breast Cancer Information: https://www.cdc.gov/cancer/breast/"
                ]
            }
        }
        self.chat_session = None
        if self.model:
            try:
                # Start a chat session for conversational context
                self.chat_session = self.model.start_chat(history=[])
            except Exception as e:
                st.warning(f"Could not start Gemini chat session: {e}")
                self.chat_session = None


    def generate_chat_response(self, user_question: str, ml_prediction_details: Dict[str, Any] = None) -> str:
        """Generate a conversational response using Gemini API, considering past interactions."""
        if not self.model:
            return "Chatbot is not available due to an initialization error."
        if not self.chat_session: # Fallback if chat session failed
            return "Chat session not available. Please try refreshing."

        try:
            prompt = user_question
            if ml_prediction_details:
                # Add context about the latest ML prediction if available and relevant to the question
                # This is a simple way; more sophisticated context injection might be needed.
                prompt = f"""
                The user has received the following breast cancer prediction results:
                - Prediction: {ml_prediction_details.get('prediction_text', 'N/A')}
                - Confidence (Malignant): {ml_prediction_details.get('confidence_malignant', 'N/A')}%
                - Confidence (Benign): {ml_prediction_details.get('confidence_benign', 'N/A')}%
                - Top Influential Features: {json.dumps(ml_prediction_details.get('top_features_dict', {}), indent=2)}
                - User's Input Data: {json.dumps(ml_prediction_details.get('input_data', {}), indent=2)}

                User's question: {user_question}

                Please answer the user's question empathetically and informatively based on this context.
                Avoid making medical diagnoses. Remind them this is not medical advice and to consult a doctor.
                If the question is unrelated to the prediction, answer generally.
                """
            
            response = self.chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            st.error(f"Chatbot error: {e}")
            return "I apologize, but I encountered an error. Please try asking in a different way or check the logs."

    def get_resources(self, risk_level: str) -> List[str]:
        """Get relevant resources based on risk level."""
        return self.knowledge_base.get(risk_level, {}).get("resources", [])

def initialize_chatbot_session():
    """Initialize the chatbot and chat history in session state."""
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = BreastCancerChatbot()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [] # Will store {"role": "user/assistant", "content": "message"}

def add_message_to_history(role: str, content: str):
    """Add a message to the chat history in session state."""
    st.session_state.chat_messages.append({"role": role, "content": content})

def display_chat_messages():
    """Display chat messages from session state."""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])