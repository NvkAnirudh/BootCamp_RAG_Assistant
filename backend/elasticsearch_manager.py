from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict, Any

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