import os
import streamlit as st
import requests
import json
import hashlib
import pickle
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Customer Service Chat with Llama-3.2-3B!",
    page_icon=":brain:",
    layout="centered",
)

# LM Studio API configuration
LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Initialize session state for chat history, user login, and context storage
if "lm_studio_history" not in st.session_state:
    st.session_state.lm_studio_history = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "context" not in st.session_state:
    st.session_state.context = {}

# Q-learning variables
learning_rate = 0.1
epsilon = 0.1
Q_table = {}

# Credentials file
CREDENTIALS_FILE = "credentials.txt"

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to save user credentials
def save_user_credentials(email, password):
    hashed_password = hash_password(password)
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            for line in f:
                if line.strip().split(',')[0] == email:
                    return False  # Email already registered
    with open(CREDENTIALS_FILE, 'a') as f:
        f.write(f"{email},{hashed_password}\n")
    return True

# Function to verify user credentials
def verify_user_credentials(email, password):
    hashed_password = hash_password(password)
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            for line in f:
                stored_email, stored_hashed_password = line.strip().split(',')
                if stored_email == email and stored_hashed_password == hashed_password:
                    return True  # Credentials are valid
    return False

# Function to save Q-table to a file
def save_q_table(user_email):
    with open(f'{user_email}_q_table.pkl', 'wb') as f:
        pickle.dump(Q_table, f)

# Function to load Q-table from a file
def load_q_table(user_email):
    global Q_table
    if os.path.exists(f'{user_email}_q_table.pkl'):
        with open(f'{user_email}_q_table.pkl', 'rb') as f:
            Q_table = pickle.load(f)
    else:
        Q_table = {}

# Function to save chat history and context in a single file
def save_chat_and_context(user_email):
    data = {
        "history": st.session_state.lm_studio_history,
        "context": st.session_state.context
    }
    with open(f'{user_email}_data.pkl', 'wb') as f:
        pickle.dump(data, f)

# Function to load chat history and context from a single file
def load_chat_and_context(user_email):
    if os.path.exists(f'{user_email}_data.pkl'):
        with open(f'{user_email}_data.pkl', 'rb') as f:
            data = pickle.load(f)
            st.session_state.lm_studio_history = data.get("history", [])
            st.session_state.context = data.get("context", {})
    else:
        st.session_state.lm_studio_history = []
        st.session_state.context = {}

# Function to store key context (e.g., laptop model)
def store_key_information(user_message, user_email):
    if "laptop model" in user_message.lower():  # Check if laptop model is mentioned
        st.session_state.context["laptop_model"] = "Latitude 7420"
        save_chat_and_context(user_email)  # Save context immediately

# Function to send a request to the LM Studio API
def get_lm_response(user_prompt):
    lm_payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [{"role": "user", "content": user_prompt}]
    }
    try:
        response = requests.post(LM_STUDIO_API_URL, json=lm_payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to LM Studio API: {e}")
        return "Sorry, I couldn't fetch a response right now."

# Function to find an existing answer for the user prompt
def find_existing_answer(user_prompt):
    for item in st.session_state.lm_studio_history:
        if item['role'] == 'user' and item['content'] == user_prompt:
            idx = st.session_state.lm_studio_history.index(item)
            if idx + 1 < len(st.session_state.lm_studio_history):
                return st.session_state.lm_studio_history[idx + 1]['content']
    return None

# Function to choose an action based on the Q-table and user input
def choose_action(state, possible_responses):
    if np.random.rand() < epsilon:
        return np.random.choice(possible_responses)
    else:
        return max(Q_table.get(state, {}), key=Q_table.get(state, {}).get, default=np.random.choice(possible_responses))

# Function to update the Q-table and store feedback
def update_q_table(state, chosen_action, feedback, user_email, user_feedback_text=None):
    reward = 1 if feedback == "positive" else -1
    Q_table.setdefault(state, {})
    
    # Update the Q-table with the feedback
    Q_table[state][chosen_action] = Q_table[state].get(chosen_action, 0) + learning_rate * (reward - Q_table[state].get(chosen_action, 0))
    
    # Store custom feedback in the vector store if needed
    if feedback == "negative" and user_feedback_text:
        store_feedback(state, user_feedback_text, user_email)
    
    save_q_table(user_email)  # Ensure Q-table is saved after update

# Function to store user feedback
def store_feedback(question, feedback_answer, user_email):
    feedback_file = f'{user_email}_feedback.txt'
    with open(feedback_file, 'a') as f:
        f.write(f"{question},{feedback_answer}\n")

# Function to respond to the user, using any previously stored context
def respond_to_user(user_message):
    if "laptop model" in user_message.lower():
        if "laptop_model" in st.session_state.context:
            st.write(f"You mentioned earlier that you're having an issue with the {st.session_state.context['laptop_model']}. How can I assist you further?")
        else:
            st.write("I don't have any details about the laptop model yet. Could you please share it again?")

# Main Streamlit app with login system
def main():
    st.title("🤖 Customer Service Chat with Llama-3.2-3B")
    
    if st.session_state.logged_in:
        st.subheader("Chat History")
        for message in st.session_state.lm_studio_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_prompt = st.chat_input("Ask Llama-3.2-3B...")

        if user_prompt:
            # Store key information like laptop model
            store_key_information(user_prompt, st.session_state.email)

            # Add user's message to history
            st.session_state.lm_studio_history.append({"role": "user", "content": user_prompt})
            st.chat_message("user").markdown(user_prompt)

            # Check for existing answer
            existing_answer = find_existing_answer(user_prompt)
            if existing_answer:
                lm_assistant_message = existing_answer
                st.write("Using cached response:")
            else:
                # Get the model's response
                with st.spinner("Fetching response..."):
                    lm_assistant_message = get_lm_response(user_prompt)

            st.session_state.lm_studio_history.append({"role": "assistant", "content": lm_assistant_message})
            with st.chat_message("assistant"):
                st.markdown(lm_assistant_message)

            # Respond based on previously stored context
            respond_to_user(user_prompt)

            # Get user feedback
            feedback = st.radio("Was this answer helpful?", ("positive", "negative"), key="feedback")
            user_feedback_text = st.text_input("Please provide feedback for the answer (if any):")
            if st.button("Submit Feedback"):
                update_q_table(user_prompt, lm_assistant_message, feedback, st.session_state.email, user_feedback_text)
                st.success("Feedback submitted successfully!")

        if st.button("Logout"):
            st.session_state.logged_in = False
            save_chat_and_context(st.session_state.email)  # Save chat and context on logout
            st.session_state.lm_studio_history = []
            st.session_state.email = None
            st.success("Logged out successfully!")

    else:
        with st.sidebar:
            email = st.text_input("Enter your email")
            password = st.text_input("Enter your password", type='password')

            if st.button("Login"):
                if verify_user_credentials(email, password):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    load_q_table(email)  # Load user's Q-table
                    load_chat_and_context(email)  # Load user's chat history and context
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid email or password!")

            if st.button("Register"):
                if save_user_credentials(email, password):
                    st.success("Registered successfully! Please log in.")
                else:
                    st.error("Email already registered. Please log in.")

if __name__ == "__main__":
    main()
