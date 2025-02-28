import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the pipeline module
from pipeline.pipeline import (
    decision_node, fetch_weather, retrieve_from_pinecone,
    general_response, get_gemini_embedding, generate_rag_response, retrieve_relevant_docs
)

### TEST CASES ###

def test_decision_node():
    """Test classification of user queries into correct categories."""
    test_cases = [
        ("What's the weather in New York?", "weather"),
        ("Tell me about quantum physics.", "general"),
        ("Summarize this document.", "rag")
    ]

    for query, expected_action in test_cases:
        state = {"user_query": query}
        result = decision_node(state)
        assert result["action"] == expected_action, f"Failed for query: {query}"

@patch("pipeline.pipeline.requests.get")
def test_fetch_weather(mock_get):
    """Test weather API response handling."""
    mock_get.return_value.json.return_value = {
        "main": {"temp": 22.5},
        "weather": [{"description": "clear sky"}]
    }

    state = {"user_query": "What is the weather in London?"}
    result = fetch_weather(state)

    assert "London" in result["response"]
    assert "clear sky" in result["response"]
    assert "22.5°C" in result["response"]

@patch("pipeline.pipeline.GoogleGenerativeAIEmbeddings.embed_query")
def test_get_gemini_embedding(mock_embed):
    """Ensure query embeddings return correct shape."""
    mock_embed.return_value = [0.1] * 768  # Mock embedding vector
    embedding = get_gemini_embedding("What is AI?")
    assert len(embedding) == 768, "Embedding shape mismatch!"

@patch("pipeline.pipeline.ChatGoogleGenerativeAI.invoke")
def test_general_response(mock_invoke):
    """Test general response generation using LLM."""
    mock_invoke.return_value = MagicMock(content="Artificial Intelligence is a branch of science.")

    state = {"user_query": "What is AI?"}
    result = general_response(state)

    assert "Artificial Intelligence" in result["response"]

@patch("pipeline.pipeline.ChatGoogleGenerativeAI.invoke")
def test_generate_rag_response(mock_invoke):
    """Test RAG-based response generation."""
    mock_invoke.return_value = MagicMock(content="AI is a field of study focusing on machine intelligence.")

    response = generate_rag_response("Tell me about AI.")

    assert "AI is a field of study" in response
