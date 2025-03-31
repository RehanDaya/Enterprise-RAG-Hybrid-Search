from agent.gemma_model import GemmaModel
import logging

class KnowledgeAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = GemmaModel()
        self.logger = logging.getLogger(__name__)
        
    def process_query(self, query):
        try:
            # Retrieve relevant documents
            retrieval_results = self.retriever.retrieve(query)
            context = retrieval_results["context"]
            documents = retrieval_results["documents"]
            
            # Prepare citations with proper filenames
            citations = []
            for i, doc in enumerate(documents):
                # Try to get original filename if available
                source = doc.metadata.get('source', 'Unknown')
                original_filename = doc.metadata.get('original_filename', source)
                
                # Use original filename in citation if available
                citation = f"[{i+1}] {original_filename}"
                citations.append(citation)
            
            # Create prompt with context for Gemma
            prompt = f"""Answer the following question based on the provided context.
            If you cannot answer based on the context, say "I don't have enough information."
            Include relevant citation numbers [N] in your answer.
            
            Context:
            {context}
            
            Question: {query}
            
            Citations:
            {chr(10).join(citations)}
            """
            
            # Get response from Gemma
            response = self.llm.invoke(prompt)
            
            self.logger.info(f"Successfully processed query: {query}")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"