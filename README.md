# Project Title & Goal
RAG-Powered Technical Documentation Assistant (Chatbot) The goal is to build an intelligent, domain-specific chatbot that can accurately answer complex technical questions by retrieving and synthesizing information from a private set of internal documentation.

# Core Feature Set & Architecture
- Document Ingestion: A Python script reads technical documents (e.g., PDFs, Markdown) and processes them into chunks using a LangChain document loader.
- Embedding & Indexing: Text chunks are converted into embeddings using an OpenAI or equivalent embedding model, and then indexed into a FAISS or Elasticsearch vector store.
- Conversational Agent: The application uses LangGraph (or advanced LangChain techniques) to manage the state of the conversation, query the vector store for context (vector search), and pass the retrieved context along with the user's prompt to the OpenAI LLM API.
- Interface: A simple web interface  to interact with the agent.



To run this complete project:

## Setup Environment:


### Create the project directory structure
mkdir -p rag_assistant/src rag_assistant/docs_source rag_assistant/vector_store rag_assistant/tests
### Change into the root directory
cd rag_assistant
### Create the .env file and paste your API key
echo "OPENAI_API_KEY='your_openai_api_key_here'" > .env 

## Install Dependencies:

pip install -r requirements.txt
(You'll need to manually paste the content from Section 1 into requirements.txt).

## Run Ingestion:

python src/ingest.py
(This creates the vector_store/faiss_index needed by the app).

## Run Tests:
python -m unittest tests.test_ingest
python -m unittest tests.test_agent


## Run Application:
streamlit run src/main_app.py
This will open the web application in your browser.
