# test_agent.py
import os
from dotenv import load_dotenv
from app.retriever.vector_store import VectorStore
from app.retriever.enhanced_retriever import EnhancedRetriever
from app.agent.knowledge_agent import KnowledgeAgent

load_dotenv()

# Load vector store
store = VectorStore()
db = store.load_existing()

# Create retriever and agent
retriever = EnhancedRetriever(db)
agent = KnowledgeAgent(retriever)

# Test query
response = agent.process_query("What are our company's core values?")
print("Response:")
print(response)