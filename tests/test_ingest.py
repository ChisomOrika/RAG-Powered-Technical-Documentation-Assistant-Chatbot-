import unittest
import os
from src.ingest import ingest_documents

class TestIngest(unittest.TestCase):
    """Tests the document ingestion process."""
    
    def setUp(self):
        # Setup dummy environment
        self.doc_dir = "test_docs"
        self.doc_file = os.path.join(self.doc_dir, "test_manual.txt")
        self.index_path = "test_vector_store/faiss_index"

        if not os.path.exists(self.doc_dir): os.makedirs(self.doc_dir)
        with open(self.doc_file, "w") as f:
            f.write("Test content 1 for ingestion. Test content 2 for ingestion.")

    def tearDown(self):
        # Cleanup dummy environment
        if os.path.exists(self.doc_file): os.remove(self.doc_file)
        if os.path.exists(self.doc_dir): os.rmdir(self.doc_dir)
        
        # Simple cleanup of FAISS files (FAISS creates multiple files)
        if os.path.exists(self.index_path + ".faiss"): os.remove(self.index_path + ".faiss")
        if os.path.exists(self.index_path + ".pkl"): os.remove(self.index_path + ".pkl")
        if os.path.exists(os.path.dirname(self.index_path)): os.rmdir(os.path.dirname(self.index_path))

    def test_ingestion_creates_index(self):
        """Verify that running ingest_documents creates the expected FAISS files."""
        ingest_documents(doc_path=self.doc_file, index_path=self.index_path)
        
        # Check for the two main FAISS files
        self.assertTrue(os.path.exists(self.index_path + ".faiss"), "FAISS index file (.faiss) was not created.")
        self.assertTrue(os.path.exists(self.index_path + ".pkl"), "FAISS index file (.pkl) was not created.")

if __name__ == '__main__':
    unittest.main()
