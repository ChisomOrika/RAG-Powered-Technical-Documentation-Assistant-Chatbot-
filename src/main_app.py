import streamlit as st
from src.agent import setup_rag_chain
import os

# --- Setup ---
st.set_page_config(page_title="RAG Documentation Assistant", layout="wide")

# Check for API key and index file before starting the app logic
if not os.getenv("OPENAI_API_KEY"):
    st.error("FATAL: OPENAI_API_KEY environment variable is not set. Please set it in your .env file or shell.")
    st.stop()
    
INDEX_PATH = "vector_store/faiss_index"
if not os.path.exists(INDEX_PATH):
    st.warning("Vector store index not found. Please run 'python src/ingest.py' first.")
    st.stop()


@st.cache_resource
def load_rag_chain():
    """Loads the RAG chain and caches it for performance."""
    try:
        return setup_rag_chain(index_path=INDEX_PATH)
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

RAG_CHAIN = load_rag_chain()

# --- Application UI ---
st.title("📚 RAG Documentation Assistant")
st.markdown("Ask a question about the technical documentation.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from RAG chain
    if RAG_CHAIN:
        with st.spinner("Searching documentation..."):
            response = RAG_CHAIN.invoke(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
