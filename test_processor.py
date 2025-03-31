# test_processor.py
from app.document_processor import DocumentProcessor

processor = DocumentProcessor()
docs = processor.load_documents("data/docs")
chunks = processor.chunk_documents(docs)
print(f"Loaded {len(chunks)} chunks from documents")