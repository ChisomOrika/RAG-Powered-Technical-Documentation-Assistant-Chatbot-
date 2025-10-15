import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

def ingest_documents(doc_path: str = "docs_source/manual.txt", index_path: str = "vector_store/faiss_index"):
    """Loads a document, chunks it, creates embeddings, and saves to FAISS."""
    
    # 1. Load Document
    print(f"--- 1. Loading Document from {doc_path}...")
    loader = TextLoader(doc_path)
    documents = loader.load()

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"--- 2. Split into {len(texts)} chunks.")

    # 3. Create Embeddings and Index
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("--- 3. Creating embeddings and FAISS index...")
    db = FAISS.from_documents(texts, embeddings)
    
    # 4. Save the Vector Store
    if not os.path.exists(os.path.dirname(index_path)):
        os.makedirs(os.path.dirname(index_path))

    db.save_local(index_path)
    print(f"--- 4. FAISS index saved to {index_path}.")

if __name__ == '__main__':
    # Ensure the docs_source directory and dummy file exist
    if not os.path.exists("docs_source"): 
        os.makedirs("docs_source")
    if not os.path.exists("docs_source/manual.txt"):
        print("Creating dummy manual.txt file...")
        with open("docs_source/manual.txt", "w") as f:
            f.write("Chapter 1: System Setup and Initialization. To successfully configure the system for the first time, you must execute the 'setup.sh' script located in the /etc/init directory. Chapter 2: Troubleshooting and Logging. If the application encounters a fatal error, you must check the logs. Application logs are stored in the persistent directory /var/log/app/service.log.")
    
    ingest_documents()
    print("\nIngestion complete. You can now run the main_app.py.")
