import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
from .document_processor import DocumentProcessor
from .elasticsearch_manager import ElasticSearchManager

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
    
    def _truncate_text(self, text: str, max_chars: int = 2000) -> str:
        """Helper function to truncate text to a maximum number of characters."""
        return text if len(text) <= max_chars else text[:max_chars] + "..."
    
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
            truncated_content = self._truncate_text(chroma_result['documents'][0], max_chars=2000)
            final_results.append({
                'content': truncated_content,
                # 'content': chroma_result['documents'][0],
                'source': chroma_result['metadatas'][0]['source'],
                'chunk_id': doc_id
            })
            
        return final_results