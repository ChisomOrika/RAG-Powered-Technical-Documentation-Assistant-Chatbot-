import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def setup_rag_chain(index_path: str = "vector_store/faiss_index"):
    """
    Sets up the LangChain RAG pipeline by loading the vector store 
    and combining it with an LLM and a prompt template.
    """
    
    # 1. Load components (embeddings model must match the one used in ingest.py)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if index exists before attempting to load
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store not found at {index_path}. Run ingest.py first!")

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 chunks
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # 2. Define Prompt
    template = """You are a helpful technical documentation assistant. Use the following context 
    to answer the user's question accurately. If you don't know the answer based on the context, 
    politely state that the information is not available in the documentation.
    
    CONTEXT: {context}
    
    QUESTION: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Build the RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def run_query(query: str, rag_chain):
    """Invokes the RAG chain with a specific query."""
    response = rag_chain.invoke(query)
    return response

if __name__ == '__main__':
    # Simple test execution
    try:
        chain = setup_rag_chain()
        print("RAG Chain initialized. Running test query...")
        test_query = "Where are the application logs stored?"
        result = run_query(test_query, chain)
        print(f"\nQUERY: {test_query}\nRESPONSE: {result}")
    except Exception as e:
        print(f"Error during agent test: {e}")
        print("Ensure 'src/ingest.py' was run successfully first.")
