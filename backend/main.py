from .hybrid_search import HybridSearchSystem
from .config import Config

def main():
    # Configuration
    PDF_DIR = Config.PDF_DIR
    CHROMA_DIR = Config.CHROMA_DIR
    
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