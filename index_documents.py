from app.document_processor import DocumentProcessor
from app.retriever.vector_store import VectorStore

# Process documents
processor = DocumentProcessor()
documents = processor.load_documents("data/docs")
chunks = processor.chunk_documents(documents)

# Create vector store
store = VectorStore()
db = store.create_from_documents(chunks)

print(f"Indexed {len(chunks)} chunks into the vector database")