import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

# Initialize LangGraph workflow
from pipeline.pipeline import graph  # Import the LangGraph pipeline

st.set_page_config(page_title="Chatbot with RAG & Weather", page_icon="🤖")

st.title("🤖 Chatbot with RAG & Weather")

# Conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Get user input
user_query = st.chat_input("Ask me anything...")

if user_query:
    # Save user query
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Run the LangGraph pipeline
    response = graph.invoke({"user_query": user_query})["response"]

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display response
    with st.chat_message("assistant"):
        st.write(response)
