# test_retrieval.py
from app.retriever.vector_store import VectorStore
from app.retriever.enhanced_retriever import EnhancedRetriever
import logging

logging.basicConfig(level=logging.INFO)

# Load vector store
store = VectorStore()
db = store.load_existing()

# Create retriever
retriever = EnhancedRetriever(db)

# List all documents in the database
all_docs = db.get()
sources = set([meta.get('source') for meta in all_docs['metadatas']])
print(f"Documents in database: {sources}")

# Test query
query = "What experience does Daya have?"
results = retriever.retrieve(query)
print(f"Query: {query}")
print(f"Retrieved sources: {[doc.metadata.get('source') for doc in results['documents']]}")