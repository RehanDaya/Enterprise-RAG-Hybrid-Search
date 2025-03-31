# reindex_all.py
import os
import logging
from app.document_processor import DocumentProcessor
from app.retriever.vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_all_documents():
    """Reindex all documents with the new embedding model"""
    logger.info("Starting reindexing process with new embedding model...")
    
    # Check if chroma_db exists and remove it
    if os.path.exists("./chroma_db"):
        logger.info("Removing existing vector database...")
        import shutil
        shutil.rmtree("./chroma_db")
    
    # Process documents
    logger.info("Processing documents...")
    processor = DocumentProcessor()
    documents = processor.load_documents("data/docs")
    chunks = processor.chunk_documents(documents)
    logger.info(f"Processed {len(chunks)} chunks from {len(documents)} documents")
    
    # Create new vector store
    logger.info("Creating new vector store with improved embeddings...")
    store = VectorStore()
    db = store.create_from_documents(chunks)
    
    logger.info(f"Successfully reindexed {len(chunks)} chunks into the vector database")
    return db

if __name__ == "__main__":
    reindex_all_documents()