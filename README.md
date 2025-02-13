# RAG-Powered Data Engineering Bootcamp Assistant

This project builds a Retrieval-Augmented Generation (RAG) system that allows users to query knowledge extracted from Zach Wilson's Free Data Engineering 
Bootcamp. This system leverages transcripts from the bootcamp videos (stored into 39 PDFs) and enables users to interact with the content using an 
OpenAI-powered (o1-mini) chatbot with contextual retrieval. The combined transcripts amount to roughly 700,000 tokens.

## Table of Contents
- [Demo](#demo)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Challenges & Next Steps](#challenges--next-steps)
- [License](#license)

## Demo
https://github.com/user-attachments/assets/b8da42ed-f437-46b7-9c3d-c169218cc15f

## Architecture Overview

```
┌───────────────────────┐         ┌───────────────────────┐
│       Streamlit       │         │     FastAPI Backend   │
│    Frontend (UI)      │◄───────►│       (/chat endpoint)│
└──────────▲────────────┘ HTTP    └──────────┬────────────┘
           │                                  │
           │                                  │
           │                                  ▼
           │                         ┌────────────────────┐
           │                         │Hybrid Search System│
           │                         │  - Reciprocal Rank │
           │                         │    Fusion Combiner │
           │                         └─────────┬──────────┘
           │                                   │
           │                                   ├──────────────────────┐
           │                                   ▼                      ▼
           │                         ┌──────────────────┐  ┌───────────────────┐
           │                         │  ChromaDB         │ │ Elasticsearch     │
           │                         │ (Vector Store)    │ │ (BM25 Index)      │
           │                         └───────┬──────────┘  └─────────┬─────────┘
           │                                   │                        │
           │                                   │                        │
           ▼                                   ▼                        ▼
┌───────────────────────┐           ┌───────────────────────┐  ┌───────────────────────┐
│   OpenAI API          │           │ Document Processing   │  │ Document Processing   │
│ (o1-mini Response)    │◄──────────┤  - PyPDF Extraction   │  │  - Sentence Tokenize  │
└───────────────────────┘           │  - NLTK Sentence Split│  │  - Sliding Window     │
                                    │  - Chunk Generation   │  │    Chunking           │
                                    └──────────┬────────────┘  └──────────┬────────────┘
                                               │                          │
                                               ▼                          ▼
                                     ┌───────────────────────┐  ┌───────────────────────┐
                                     │  Sentence Transformer │  │  39 PDF Transcripts   │
                                     │ (all-MiniLM-L6-v2)    │  │ (Source Documents)    │
                                     └───────────────────────┘  └───────────────────────┘
```

1. **Document Processing**  
   - **Extraction & Chunking:**  
     The `DocumentProcessor` extracts text from 39 PDF transcripts using `pypdf` and tokenizes the text into sentences using NLTK. It then splits the
     transcripts into contextual chunks (with a sliding window of sentences) to preserve context.

2. **Indexing**  
   - **Elasticsearch (BM25 Search):**  
     The `ElasticSearchManager` creates an index (if it doesn’t exist) and bulk-indexes each chunk. This enables keyword-based BM25 retrieval.
   - **ChromaDB (Vector-based Semantic Search):**  
     The system uses ChromaDB to store embeddings of the chunks (generated using `SentenceTransformer` with the `all-MiniLM-L6-v2` model). This allows
     for semantic search over the bootcamp content by comparing query embeddings with those stored in the vector database.

3. **Hybrid Retrieval & RAG**  
   - The `HybridSearchSystem` performs two types of search:
     - **Semantic Search with ChromaDB:** Queries the vector store to retrieve context based on similarity.
     - **Keyword-Based Search with Elasticsearch (BM25):** Retrieves documents using traditional keyword matching.
   - A reciprocal rank fusion method combines the results from both sources to select the most relevant chunk
   - The selected context is then passed, along with a system prompt and the user's query, to the OpenAI API to generate a final answer.

4. **Backend API**  
   - Built with FastAPI, the backend exposes a `/chat` endpoint that handles incoming queries.
   - The endpoint retrieves relevant context from both Elasticsearch and ChromaDB, constructs the prompt, and generates the response using the OpenAI
     client.

5. **Frontend Interface**  
   - The Streamlit frontend (`frontend/app.py`) provides an interactive chat interface.
   - Users can input queries, view conversation history, and inspect the source documents used to generate responses.

## Project Structure
```plaintext
├── backend
│   ├── api.py               # FastAPI endpoint for chat
│   ├── config.py            # Configuration and environment variable loading
│   ├── document_processing.py  # PDF extraction and contextual chunking
│   ├── elasticsearch_manager.py # BM25 search via Elasticsearch
│   ├── hybrid_search.py     # Combines semantic and BM25 search results
│   ├── llm_integration.py   # OpenAI API integration using o1-mini model
│   └── main.py              # Script for indexing documents and testing queries
├── frontend
│   └── app.py               # Streamlit app for the chat interface
├── setup_environment.sh     # Script to set up Python virtual environment
├── .env                     # Environment variables (API keys)
└── requirements.txt         # Python package dependencies
```

## Setup Instructions

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Set Up the Environment** - Run the provided setup script:
   
   ```
   bash setup_environment.sh
   ```
   This script creates a virtual environment, installs dependencies, and downloads NLTK.
3. **Configure Environment Variables**
   - Create a ```.env``` file in the root directory.
   - Add your OpenAI API key:

     ```
     OPENAI_API_KEY=your-api-key-here
     ```
4. **Start Elasticsearch** - You can run Elasticsearch locally using Docker:

   ```
   docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
   ```
## Running the Project
### Run the Backend
From the root directory, run:
```
uvicorn backend.api:app --reload
```
Note: The --reload flag is required during development so that the server automatically restarts on code changes.
### Run the Frontend
Open a new terminal and run:
```
streamlit run frontend/app.py
```
## Usage
1. **Indexing Documents:**
   Before using the chat interface, ensure that the PDFs are processed and indexed. Run the following command to process all 39 PDF transcripts,
   create contextual chunks, index them in Elasticsearch, and generate embeddings for ChromaDB:
   ```
   python backend/main.py
   ```
2. **Interacting via the Chat Interface:**
   Open the Streamlit app in your browser. Type your query into the chat input. The backend retrieves the most relevant context from both Elasticsearch
   and ChromaDB and passes it along with the user query to the OpenAI API. The chatbot's response, along with the sources used, will be displayed in
   the interface.
## Challenges & Next Steps
### Challenges
- **Handling Large Contexts:**
With a total token count of 700,000 across all transcripts, ensuring that the LLM receives relevant context without overwhelming it was challenging.
Initially, using too many tokens (e.g., 80k tokens) led to high inference times.
- **Cost and Model Limitations:**
The project uses the cost-effective `o1-mini-2024-09-12` model to keep expenses low while still providing meaningful responses. Unlike larger models
such as GPT-3.5 or GPT-4, the `o1-mini` model was chosen specifically for its strong reasoning capabilities relative to its cost. It offers a good
balance between performance and affordability, making it ideal for processing large contexts (from the 39 bootcamp PDFs) without incurring high API
costs. This strategic choice allows the system to deliver insightful answers while efficiently managing operational expenses.

### Next Steps
- **Implement Re-Ranking:**
Add a re-ranking step to further refine and reduce the context by selecting only the most relevant chunks from an initially larger set of retrieved
documents.
- **Optimize API Calls:**
Incorporate streaming responses and parallelize retrieval from Elasticsearch and ChromaDB to reduce perceived latency.
- **Improve Chunking Strategy:**
Experiment with different chunk sizes and summarization techniques to balance context richness with inference performance.
- **Deploy and Scale:**
Package the application into Docker containers and deploy it on a cloud platform for production use.
- **User Authentication:**
Consider adding authentication to manage access, especially if you plan to deploy the application publicly.

## License
This project is licensed under the MIT License.

   
   
   
