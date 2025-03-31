# vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import logging
import shutil
import time

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Use the stronger embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )
        self.db = None
        self.logger = logging.getLogger(__name__)

    def create_from_documents(self, documents):
        # If no documents are provided, create an empty database
        if not documents:
            self.logger.info("Creating empty database...")
            try:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                # Attempt to delete all existing documents
                all_ids = self.db._collection.get()["ids"]
                if all_ids:
                    self.db._collection.delete(all_ids)
                self.logger.info(f"Created empty database and deleted {len(all_ids)} documents")
                return self.db
            except Exception as e:
                self.logger.error(f"Error creating empty database: {str(e)}")
                raise e
            
        # Normal creation with documents
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        return self.db

    def hard_reset(self):
        """Force a complete reset of the database by deleting all content via API"""
        self.logger.info("Performing API-based reset of the database...")
        
        try:
            # Load the existing database if not already loaded
            if self.db is None:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
            
            # Get all document IDs in the database
            all_data = self.db.get()
            
            if all_data and 'ids' in all_data and all_data['ids']:
                # Delete all documents using the API instead of deleting files
                self.logger.info(f"Deleting {len(all_data['ids'])} documents from database...")
                self.db.delete(all_data['ids'])
                self.logger.info("Successfully deleted all documents from database")
            else:
                self.logger.info("Database is already empty")
                
            # The database is now empty but still exists on disk
            return True, self.db
            
        except Exception as e:
            self.logger.error(f"Error during API-based reset: {str(e)}")
            
            # Try to create a fresh connection as fallback
            try:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
                return False, self.db
            except:
                raise e

    def load_existing(self):
        if os.path.exists(self.persist_directory):
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            return self.db
        else:
            raise FileNotFoundError(f"No existing database found at {self.persist_directory}")
            
    def add_documents(self, documents):
        if self.db is None:
            return self.create_from_documents(documents)
        else:
            self.db.add_documents(documents)
            return self.db
    
    def delete_document(self, document_name):
        """Remove all chunks associated with a specific document"""
        if self.db is None:
            return False, 0
        
        try:
            # Get IDs of all chunks with the given source
            results = self.db.get(where={"source": document_name})
            if results and 'ids' in results and len(results['ids']) > 0:
                # Delete the chunks
                self.db.delete(ids=results['ids'])
                self.logger.info(f"Deleted {len(results['ids'])} chunks from document: {document_name}")
                return True, len(results['ids'])
            return False, 0
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False, 0
    
    def list_documents(self):
        """List all unique document sources in the database"""
        if self.db is None:
            return {}
        
        try:
            all_docs = self.db.get()
            if all_docs and 'metadatas' in all_docs and len(all_docs['metadatas']) > 0:
                # Extract unique source names
                sources = {}
                for i, meta in enumerate(all_docs['metadatas']):
                    source = meta.get('source', 'Unknown')
                    if source in sources:
                        sources[source] += 1
                    else:
                        sources[source] = 1
                
                return sources
            return {}
        except Exception as e:
            self.logger.error(f"Error listing documents: {str(e)}")
            return {}
            
    def get(self):
        """Get all documents from the database"""
        if self.db is None:
            return None
        try:
            return self.db.get()
        except:
            return None
            
    def similarity_search(self, query, k=5, filter=None):
        """Wrapper for similarity search"""
        if self.db is None:
            return []
        return self.db.similarity_search(query, k=k, filter=filter)
        
    def max_marginal_relevance_search(self, query, k=3, filter=None):
        """Wrapper for MMR search"""
        if self.db is None:
            return []
        return self.db.max_marginal_relevance_search(query, k=k, filter=filter)