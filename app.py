import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Streamlit app setup
st.title("AI Chatbot")
st.write("I'm a chatbot specialized in answering questions about Artificial Intelligence. Feel free to ask anything about AI!")

# Initialize session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None
if "past_inputs" not in st.session_state:
    st.session_state["past_inputs"] = []
if "past_responses" not in st.session_state:
    st.session_state["past_responses"] = []

# Input box for user query
user_input = st.text_input("You:", "")

if user_input:
    # Encode user input and append to chat history if exists
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = (
        new_input_ids
        if st.session_state["chat_history_ids"] is None
        else torch.cat([st.session_state["chat_history_ids"], new_input_ids], dim=-1)
    )

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
        num_return_sequences=1
    )

    # Decode and display the response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state["past_inputs"].append(user_input)
    st.session_state["past_responses"].append(bot_response)
    st.session_state["chat_history_ids"] = chat_history_ids

    # Display chat history
    for user, bot in zip(st.session_state["past_inputs"], st.session_state["past_responses"]):
        st.write(f"You: {user}")
        st.write(f"Bot: {bot}")

