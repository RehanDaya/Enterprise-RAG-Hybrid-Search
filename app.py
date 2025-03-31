import streamlit as st
import os
import tempfile
import sys
import logging
import warnings
import shutil
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.vector_store import VectorStore
from retriever.enhanced_retriever import EnhancedRetriever
from agent.knowledge_agent import KnowledgeAgent
from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def initialize_system():
    try:
        # Load vector store
        store = VectorStore()
        try:
            db = store.load_existing()
        except FileNotFoundError:
            st.warning("No existing vector database found. Please upload documents to get started.")
            db = None
        
        # Create processor
        processor = DocumentProcessor()
        
        # Create retriever and agent only if we have a database
        retriever = None
        agent = None
        if db is not None:
            retriever = EnhancedRetriever(db)
            agent = KnowledgeAgent(retriever)
        
        return {
            "vector_store": store,
            "db": db,
            "retriever": retriever,
            "agent": agent,
            "processor": processor
        }
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

# Main app
st.title("Enterprise Knowledge Assistant")
st.caption("Powered by Gemma 7B and RAG")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "document_mapping" not in st.session_state:
    # Maps temporary filenames to original filenames
    st.session_state.document_mapping = {}

# Initialize system
system = initialize_system()

# Document management section in sidebar
with st.sidebar:
    st.header("Document Management")
    
    # Document upload section
    st.subheader("Upload Documents")
    # Use a dynamic key to force the file uploader to reset
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    uploaded_file = st.file_uploader("Upload a document to the knowledge base", 
                                    type=["pdf", "docx", "txt"],
                                    key=f"file_uploader_{st.session_state.file_uploader_key}")
    
    # Document deletion section
    st.subheader("Manage Documents")
    
    # List available documents
    if system and system["vector_store"] and system["db"]:
        document_list = system["vector_store"].list_documents()
        
        # Filter and map temporary filenames to original filenames if needed
        cleaned_doc_list = {}
        for doc_name, count in document_list.items():
            # Use original filename if available in mapping
            display_name = doc_name
            if doc_name.startswith("tmp") and any(tmp for tmp in st.session_state.document_mapping if tmp in doc_name):
                # Find the matching temp filename in our mapping
                for temp_name, original_name in st.session_state.document_mapping.items():
                    if temp_name in doc_name:
                        display_name = original_name
                        break
            
            cleaned_doc_list[display_name] = count
        
        if cleaned_doc_list:
            # Show document count
            total_docs = sum(cleaned_doc_list.values())
            st.write(f"Total chunks in database: {total_docs}")
            
            # Create a selectbox with document names and chunk counts
            doc_options = [f"{doc} ({count} chunks)" for doc, count in cleaned_doc_list.items()]
            
            if doc_options:
                selected_doc = st.selectbox("Select document to delete:", doc_options)
                
                if selected_doc:
                    # Extract document name from the selection
                    doc_name = selected_doc.split(" (")[0]
                    
                    # Find the actual filename in the database (might be a temp name)
                    actual_doc_name = doc_name
                    for temp_name, orig_name in st.session_state.document_mapping.items():
                        if orig_name == doc_name:
                            # Search for the temp name in document_list
                            matching_keys = [k for k in document_list.keys() if temp_name in k]
                            if matching_keys:
                                actual_doc_name = matching_keys[0]
                                break
                    
                    if st.button(f"Delete {doc_name}"):
                        success, count = system["vector_store"].delete_document(actual_doc_name)
                        if success:
                            st.success(f"Successfully deleted {count} chunks from {doc_name}")
                            
                            # Remove from document mapping if it exists
                            keys_to_remove = []
                            for temp_name, orig_name in st.session_state.document_mapping.items():
                                if orig_name == doc_name:
                                    keys_to_remove.append(temp_name)
                            
                            for key in keys_to_remove:
                                del st.session_state.document_mapping[key]
                            
                            # Clear Streamlit's file uploader state by forcing a key change
                            if "file_uploader_key" not in st.session_state:
                                st.session_state.file_uploader_key = 0
                            st.session_state.file_uploader_key += 1
                            
                            # Reinitialize system components with updated DB
                            system["retriever"] = EnhancedRetriever(system["db"])
                            system["agent"] = KnowledgeAgent(system["retriever"])
                            
                            # Clear cache to force reload on next page refresh
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc_name}")
            
            # Add Delete All Documents button
            st.subheader("Delete All Documents")
            if st.button("Delete All Documents", type="primary", help="This will delete all documents from the knowledge base"):
                try:
                    # Release all references first
                    system["db"] = None
                    system["retriever"] = None
                    system["agent"] = None
                    
                    # Clean up memory
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Force Streamlit to release cached resources
                    st.cache_resource.clear()
                    
                    # Create a fresh VectorStore instance
                    fresh_store = VectorStore()
                    
                    # Use the hard reset method
                    success, fresh_db = fresh_store.hard_reset()
                    
                    if success:
                        # Update system with new components
                        system["vector_store"] = fresh_store
                        system["db"] = fresh_db
                        system["retriever"] = EnhancedRetriever(fresh_db)
                        system["agent"] = KnowledgeAgent(system["retriever"])
                        
                        # Clear document mapping
                        st.session_state.document_mapping = {}
                        
                        # Clear Streamlit's file uploader state by forcing a key change
                        if "file_uploader_key" not in st.session_state:
                            st.session_state.file_uploader_key = 0
                        st.session_state.file_uploader_key += 1
                        
                        st.success("Successfully reset the knowledge base! All documents have been removed.")
                    else:
                        st.warning("Database files could not be completely removed, but the database has been reset.")
                    
                    # Force a rerun to refresh everything
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting knowledge base: {str(e)}")
                    st.error("Please try closing and reopening the app to reset completely.")
        else:
            st.info("No documents found in the database")
        
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
                temp_filename = os.path.basename(temp_file_path)
            
            # Store the mapping between temp filename and original filename
            st.session_state.document_mapping[temp_filename] = uploaded_file.name
            
            # Process the document
            processor = system["processor"]
            store = system["vector_store"]
            
            # Load and chunk the document with original filename
            chunks = processor.process_file(temp_file_path, original_filename=uploaded_file.name)
            
            # Update metadata to use original filename
            for chunk in chunks:
                # Keep the temp filename as source for deletion to work properly
                chunk.metadata['original_filename'] = uploaded_file.name
            
            # After adding to vector store
            db = store.add_documents(chunks)
            print(f"Vector DB status: {db._collection.count()} documents total")

            # Update system with new db
            system["db"] = db
            system["retriever"] = EnhancedRetriever(db)  # Always recreate
            system["agent"] = KnowledgeAgent(system["retriever"])  # Always recreate
            
            # Set system as initialized
            st.session_state.system_initialized = True
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            st.success(f"Added {len(chunks)} chunks from {uploaded_file.name} to the knowledge base")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if system is initialized before allowing chat
if not system or not system["agent"]:
    if not st.session_state.system_initialized:
        st.info("Please upload documents to initialize the knowledge base.")
else:
    # Get user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking (this may take a moment)..."):
                print(f"Query: {prompt}")
                print(f"System state: {system['db'] is not None}")
                print(f"Using retriever with {system['retriever'].vector_store._collection.count()} documents")
                response = system["agent"].process_query(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})