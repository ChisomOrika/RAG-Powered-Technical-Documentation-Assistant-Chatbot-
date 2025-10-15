To run this complete project:

Setup Environment:


# Create the project directory structure
mkdir -p rag_assistant/src rag_assistant/docs_source rag_assistant/vector_store rag_assistant/tests
# Change into the root directory
cd rag_assistant
# Create the .env file and paste your API key
echo "OPENAI_API_KEY='your_openai_api_key_here'" > .env 

### Install Dependencies:

pip install -r requirements.txt
(You'll need to manually paste the content from Section 1 into requirements.txt).

### Run Ingestion:

python src/ingest.py
(This creates the vector_store/faiss_index needed by the app).

Run Tests:

python -m unittest tests.test_ingest
python -m unittest tests.test_agent


Run Application:


streamlit run src/main_app.py
This will open the web application in your browser.
