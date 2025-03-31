from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents(self, directory_path):
        documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            else:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = filename
                doc.metadata['file_path'] = file_path
                doc.metadata['original_filename'] = filename  # Add original filename
            
            documents.extend(docs)
        return documents

    def chunk_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            
            # Ensure original filename is preserved in chunks
            if 'original_filename' not in chunk.metadata and 'source' in chunk.metadata:
                chunk.metadata['original_filename'] = chunk.metadata['source']
                
        return chunks
        
    def process_file(self, file_path, original_filename=None):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
            
        documents = loader.load()
        filename = os.path.basename(file_path)
        
        # Use the provided original filename if available
        for doc in documents:
            doc.metadata['source'] = filename
            doc.metadata['file_path'] = file_path
            if original_filename:
                doc.metadata['original_filename'] = original_filename
            else:
                doc.metadata['original_filename'] = filename
        
        chunks = self.chunk_documents(documents)
        return chunks