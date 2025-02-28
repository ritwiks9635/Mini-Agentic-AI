import os
import json
import time
import requests
import numpy as np
import langchain
import langgraph
from dotenv import load_dotenv
from langgraph.graph import Graph
from langgraph.graph import StateGraph

from langsmith import traceable, Client
from pydantic import BaseModel, Field

from typing import TypedDict, Optional
from langchain.prompts import PromptTemplate

import google.generativeai as genai

from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader

# Import the required classes directly
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

index_name = 'qa-bot'
PINE_API_KEY = os.getenv("PINECONE_API_KEY")
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key=PINE_API_KEY)
index = pc.Index(index_name)

class PipelineState(TypedDict):
    user_query: str
    action: Optional[str]
    response: Optional[str]

def decision_node(state: PipelineState):
    query = state["user_query"]

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    classification_prompt = f"""
    Classify this query as either 'weather', 'rag', or 'general'.
    - If it is asking about weather conditions, classify as 'weather'.
    - If it is asking about information from a provided document, classify as 'rag'.
    - If the query does not fit into these categories, classify as 'general'.

    Query: {query}
    """

    action = llm.invoke(classification_prompt)

    # Ensure action is valid
    action_str = action.content.lower()
    if action_str not in ["weather", "rag", "general"]:
        action_str = "general"
    else:
        action_str = action.content.lower()

    return {"action": action_str}

def fetch_weather(state: PipelineState):
    user_query = state["user_query"]

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    # Prompt template for extracting city name
    city_extraction_prompt = PromptTemplate.from_template(
        "Extract the city name from this query: {query}. If no city is found, return 'None'."
    )
    # Ask Gemini LLM to extract the city name
    prompt = city_extraction_prompt.format(query=user_query)
    city = llm.invoke(prompt)

    if city.content.lower() == "none":
        return {"response": "I couldn't determine the city name. Please specify a valid location."}

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city.content}&appid={api_key}&units=metric"

    response = requests.get(url).json()

    if response.get("main"):
        weather_info = f"Current weather in {city.content}: {response['weather'][0]['description']}, Temperature: {response['main']['temp']}°C."
    else:
        weather_info = "Weather data not available. Please check the city name."

    return {"response": weather_info}


def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    return docs

def get_gemini_embeddings(texts, batch_size = 10):
    model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        try:
            response = model.embed_documents(batch)
            batch_embeddings = [item for item in response]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            time.sleep(5)
    return np.array(embeddings)


def store_in_pinecone(docs):
    texts = [doc.page_content for doc in docs]
    embeddings = get_gemini_embeddings(texts)

    # Store vectors in Pinecone
    for i, (text, vector) in enumerate(zip(texts, embeddings)):
        index.upsert([(str(i), vector.tolist(), {"text": text})])
    print("Stored documents in Pinecone successfully!")

def get_gemini_embedding(query):
    """Generate embeddings for the user query."""
    model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_query")
    embedding = model.embed_query(query)
    return np.array(embedding)

def retrieve_relevant_docs(query, top_k=3):
    """Retrieve the most relevant documents from Pinecone."""
    query_vector = get_gemini_embedding(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract relevant texts
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]

    return retrieved_texts


def generate_rag_response(query):
    """Use RAG to fetch relevant documents and generate an answer."""
    retrieved_docs = retrieve_relevant_docs(query)
    docs = "\n\n".join(retrieved_docs)

    # Generate response using Gemini LLM
    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
    Use the following source documents to answer the user's questions.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Documents:
    {docs}"""

    llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    response = llm_model.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": query},
        ],
    )
    return response.content

def retrieve_from_pinecone(state: PipelineState):
    query = state["user_query"]

    # Load Pinecone as the retriever
    response = generate_rag_response(query)

    return {"response": response}

def general_response(state: PipelineState):
    query = state["user_query"]

    instructions = f"""
    You are a helpful assistant, as a professional expert,
    provide a concise and informative answer to the following question,
    ensuring accuracy and addressing all relevant aspects, even if the
    question appears broad or open-ended. Answer the following Question concisely.
    """

    llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    response = llm_model.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": query},
        ],
    )
    return {"response": response.content}


# Create LangGraph workflow
workflow = StateGraph(PipelineState)

# Add nodes
workflow.add_node("decision", decision_node)
workflow.add_node("weather", fetch_weather)
workflow.add_node("rag", retrieve_from_pinecone)
workflow.add_node("general", general_response)

# Define edges (execution flow)
workflow.set_entry_point("decision")
workflow.add_conditional_edges(
    "decision",
    lambda state: state["action"],
    {
        "weather": "weather",
        "rag": "rag",
        "general": "general",
    }
)

# Finalize the graph
graph = workflow.compile()
