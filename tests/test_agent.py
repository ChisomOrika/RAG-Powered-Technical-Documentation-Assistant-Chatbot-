import unittest
import os
from src.agent import setup_rag_chain, run_query
from src.ingest import ingest_documents # Import ingest to ensure setup

class TestAgent(unittest.TestCase):
    """Integration test for the RAG chain functionality."""
    
    def setUp(self):
        # Ensure the ingestion runs before the agent test
        self.index_path = "test_vector_store/faiss_index_agent"
        
        # Create a specific dummy file for context validation
        doc_dir = "test_docs_agent"
        doc_file = os.path.join(doc_dir, "test_agent_manual.txt")
        if not os.path.exists(doc_dir): os.makedirs(doc_dir)
        with open(doc_file, "w") as f:
            f.write("The primary color of the application dashboard is blue, while all alert messages use the color red for immediate visibility.")
        
        ingest_documents(doc_path=doc_file, index_path=self.index_path)
        
        # Setup the chain to be tested
        self.rag_chain = setup_rag_chain(index_path=self.index_path)

    def tearDown(self):
        # Simple cleanup of FAISS files and directories
        if os.path.exists(self.index_path + ".faiss"): os.remove(self.index_path + ".faiss")
        if os.path.exists(self.index_path + ".pkl"): os.remove(self.index_path + ".pkl")
        if os.path.exists(os.path.dirname(self.index_path)): os.rmdir(os.path.dirname(self.index_path))
        
        doc_dir = "test_docs_agent"
        doc_file = os.path.join(doc_dir, "test_agent_manual.txt")
        if os.path.exists(doc_file): os.remove(doc_file)
        if os.path.exists(doc_dir): os.rmdir(doc_dir)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "Skipping LLM test, OPENAI_API_KEY not set.")
    def test_rag_retrieves_correct_answer(self):
        """Verifies the RAG chain uses the context to answer a question."""
        query = "What color are the alert messages?"
        response = run_query(query, self.rag_chain)
        
        # The response should explicitly contain 'red' based on the context
        self.assertIn("red", response.lower(), "The RAG chain failed to retrieve the specific color 'red' from the context.")

if __name__ == '__main__':
    unittest.main()
