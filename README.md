# **Mini-Agentic-AI**

[Loom Video](https://www.loom.com/share/343ec0d686e54b30a592c46d728ade2a)

This project is an AI-powered query pipeline using **LangChain, Google Gemini AI, Pinecone, and Streamlit** for an interactive query-handling web app.  

## **🚀 Features**  

✅ **Query Classification** – Automatically classifies queries into "weather", "rag", or "general".  

✅ **Weather API Integration** – Retrieves real-time weather information using OpenWeather API.  

✅ **Document Retrieval (RAG)** – Uses **Pinecone** and **Google Generative AI embeddings** to retrieve relevant documents for answering queries.  

✅ **General Query Handling** – Uses Gemini AI to answer open-ended questions.  

✅ **LangGraph Workflow** – Implements a **decision-based pipeline** for intelligent query handling.  

✅ **Langsmith Workflow** - Implement Langsmith for evaluate LLM model performance.   

✅ **Test Cases** - Implement Pytest workflow for test all Function poperly work or not.   

✅ **Streamlit UI** – Provides a simple, interactive web app for query input and responses.  

---

## **🛠️ Setup Instructions**  

### **1️⃣ Clone the Repository**  

```bash
git clone https://github.com/ritwiks9635/Mini-Agentic-AI.git
cd Mini-Agentic-AI
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**  

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**  

```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**  

1. **Create a `.env` file** in the project root and add the following keys:  

   ```env
   OPENWEATHER_API_KEY=your_openweather_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   GOOGLE_GEMINI_API_KEY=your_gemini_api_key
   ```

2. **Ensure `.env` is added to `.gitignore`** to prevent pushing sensitive information.  

---

## **🔧 Implementation Details**  

### **📂 Project Structure**  

```
📦 project-root
│── 📂 pipeline
│   ├── pipeline.py   # Main LangGraph workflow
├── app.py        # Streamlit UI
│── .env              # Stores API keys (not pushed to GitHub)
│── .gitignore        # Ensures .env is ignored
│── requirements.txt  # Dependencies
│── README.md         # Project documentation
```

### **🧩 How It Works**  

1. **Query is classified** using Gemini AI into:
   - `weather` (fetches weather data)
   - `rag` (retrieves relevant documents from Pinecone)
   - `general` (answers the question directly using Gemini AI)  

2. **Weather Queries** – Uses OpenWeather API to fetch real-time weather information.  

3. **RAG Queries** –  
   - Loads **PDF documents** using `PyPDFLoader`.  
   - Splits text into chunks using `RecursiveCharacterTextSplitter`.  
   - Stores embeddings in **Pinecone**.  
   - Retrieves relevant documents and generates an answer.  

4. **General Queries** – Uses Gemini AI to generate informative responses.  

5. **Streamlit Web App (`app.py`)**  
   - Users enter queries via a simple UI.  
   - Displays query classification and response.  

---

## **🖥️ Running the Project**  

### **Run the LangChain Pipeline**  

```bash
python pipeline/pipeline.py
```

### **Run the Streamlit Web App**  

```bash
streamlit run app.py
```

Then, open the link displayed in the terminal to access the app.  

---

## **📌 Future Enhancements**  
- Add a chatbot UI with Streamlit.  
- Support for voice queries.  
- Multi-document support.  

---

## **📜 License**  

This project is licensed under the **MIT License**.  
