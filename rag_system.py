import os
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import chromadb
import pypdf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import time

class DocumentProcessor:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + ' '
        return text.strip()
    
    def create_contextual_chunks(self, text: str, chunk_size: int = 512) -> List[Dict[str, str]]:
        """Create chunks with surrounding context."""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            # Create main chunk
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join(chunk_sentences)
            
            # Create context (previous and next sentences)
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(sentences), i + chunk_size + self.window_size)
            context_sentences = sentences[start_idx:i] + sentences[i + chunk_size:end_idx]
            context = ' '.join(context_sentences)
            
            chunks.append({
                'content': chunk_text,
                'context': context
            })
        
        return chunks

    def process_documents(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Process all PDFs with context."""
        all_documents = []
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        if len(pdf_files) == 0:
            raise ValueError(f"No PDF files found in directory: {pdf_dir}")
            
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            print(f"Extracted {len(text)} characters from {pdf_file.name}")
            
            if not text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue
                
            chunks = self.create_contextual_chunks(text)
            print(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            for i, chunk in enumerate(chunks):
                all_documents.append({
                    'source': pdf_file.name,
                    'chunk_id': f"{pdf_file.stem}_chunk_{i}",
                    'content': chunk['content'],
                    'context': chunk['context']
                })
        
        print(f"Total documents processed: {len(all_documents)}")
        return all_documents

class ElasticSearchManager:
    def __init__(self, index_name: str = "documents"):
        self.es = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        
    def create_index(self):
        """Create Elasticsearch index with appropriate mappings."""
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "context": {"type": "text"},
                        "chunk_id": {"type": "keyword"},
                        "source": {"type": "keyword"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in Elasticsearch."""
        actions = [
            {
                "_index": self.index_name,
                "_id": doc["chunk_id"],
                "_source": {
                    "content": doc["content"],
                    "context": doc["context"],
                    "chunk_id": doc["chunk_id"],
                    "source": doc["source"]
                }
            }
            for doc in documents
        ]
        bulk(self.es, actions)

    def search(self, query: str, size: int = 20) -> List[Dict[str, Any]]:
        """Search using BM25 on both content and context."""
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "context"],
                        "type": "best_fields"
                    }
                },
                "size": size
            }
        )
        return response["hits"]["hits"]

class HybridSearchSystem:
    def __init__(self, pdf_dir: str, chroma_dir: str = "./chroma_db"):
        self.doc_processor = DocumentProcessor()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.es_manager = ElasticSearchManager()
        
    def index_documents(self, pdf_dir: str):
        """Process and index documents in both ChromaDB and Elasticsearch."""
        print(f"Starting document processing from directory: {pdf_dir}")
        
        # Ensure the PDF directory exists
        if not os.path.exists(pdf_dir):
            raise ValueError(f"PDF directory does not exist: {pdf_dir}")
        
        # Process documents
        documents = self.doc_processor.process_documents(pdf_dir)
        
        if not documents:
            raise ValueError("No documents were processed successfully")
        
        print(f"Successfully processed {len(documents)} document chunks")
        
        # Index in Elasticsearch
        print("Indexing in Elasticsearch...")
        self.es_manager.create_index()
        self.es_manager.index_documents(documents)
        print("Elasticsearch indexing complete")
        
        # Generate embeddings and index in ChromaDB
        print("Generating embeddings...")
        texts = [doc["content"] + " " + doc["context"] for doc in documents]
        print(f"Preparing to encode {len(texts)} texts")
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )
        print(f"Generated {len(embeddings)} embeddings")
        
        print("Indexing in ChromaDB...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=[doc["content"] for doc in documents],
            metadatas=[{"source": doc["source"], "chunk_id": doc["chunk_id"]} 
                    for doc in documents],
            ids=[doc["chunk_id"] for doc in documents]
        )
        print("ChromaDB indexing complete")
        
    def reciprocal_rank_fusion(self, 
                              semantic_results: List[Dict], 
                              bm25_results: List[Dict],
                              semantic_weight: float = 0.8,
                              bm25_weight: float = 0.2) -> List[str]:
        """Merge results using reciprocal rank fusion with weights."""
        scores = {}
        
        # Process semantic search results
        for rank, (doc_id, score) in enumerate(zip(
            semantic_results['ids'][0], 
            semantic_results['distances'][0]
        )):
            scores[doc_id] = scores.get(doc_id, 0) + (semantic_weight / (rank + 1))
            
        # Process BM25 results
        for rank, hit in enumerate(bm25_results):
            doc_id = hit["_source"]["chunk_id"]
            scores[doc_id] = scores.get(doc_id, 0) + (bm25_weight / (rank + 1))
            
        # Sort by final scores
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search using both semantic search and BM25."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Get results from both systems
        semantic_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=20
        )
        
        bm25_results = self.es_manager.search(query, size=20)
        
        # Merge results
        merged_ids = self.reciprocal_rank_fusion(semantic_results, bm25_results)
        
        # Return top k results
        final_results = []
        for doc_id in merged_ids[:k]:
            chroma_result = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            final_results.append({
                'content': chroma_result['documents'][0],
                'source': chroma_result['metadatas'][0]['source'],
                'chunk_id': doc_id
            })
            
        return final_results

def main():
    # Configuration
    PDF_DIR = "./data"
    CHROMA_DIR = "./chroma_db"
    
    print("Starting initialization...")
    try:
        # Initialize system
        system = HybridSearchSystem(PDF_DIR, CHROMA_DIR)
        print("System initialized successfully")
        
        print("Checking Elasticsearch connection...")
        if not system.es_manager.es.ping():
            raise Exception("Cannot connect to Elasticsearch")
        print("Elasticsearch connection successful")
        
        print("Starting document indexing...")
        # Index documents (run only once)
        system.index_documents(PDF_DIR)
        print("Document indexing completed")
        
        # Example query
        query = "What is the main topic of the documents?"
        print(f"Running query: {query}")
        results = system.search(query)
        
        # Print results
        print("\nQuery Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {result['source']}")
            print(f"Content: {result['content'][:200]}...")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()