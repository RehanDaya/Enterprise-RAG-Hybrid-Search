# enhanced_retriever.py
import logging
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from agent.gemma_model import GemmaModel

class EnhancedRetriever:
    def __init__(self, vector_store, similarity_k=5, mmr_k=3):
        self.vector_store = vector_store
        self.similarity_k = similarity_k
        self.mmr_k = mmr_k
        self.logger = logging.getLogger(__name__)
        self.llm = GemmaModel()
        
        # Initialize reranker
        self.has_reranker = False
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.has_reranker = True
            self.logger.info("Initialized cross-encoder reranker")
        except Exception as e:
            self.logger.warning(f"Could not initialize reranker: {str(e)}")
            self.has_reranker = False
        
        # Initialize BM25 retriever for keyword search
        self.has_bm25 = False
        try:
            all_docs = self.vector_store.get()
            if all_docs and 'documents' in all_docs and len(all_docs['documents']) > 0:
                texts = all_docs['documents']
                self.bm25_retriever = BM25Retriever.from_texts(texts)
                self.has_bm25 = True
                self.logger.info(f"Initialized BM25 retriever with {len(texts)} documents")
            else:
                self.logger.warning("No documents found for BM25 retriever")
        except Exception as e:
            self.logger.warning(f"Could not initialize BM25 retriever: {str(e)}")

    def expand_query(self, query):
        """Expand the query using the LLM to improve retrieval"""
        self.logger.info(f"Expanding query: {query}")
        try:
            prompt = f"""Given the user query: "{query}"
            Generate 3-5 alternative ways to phrase this query to improve document retrieval.
            Focus on expanding with relevant professional and technical terms.
            Return only the expanded queries, one per line, no numbering or other text."""
            
            response = self.llm.invoke(prompt)
            expanded_queries = [query]
            if response and hasattr(response, 'content'):
                expansion_lines = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
                expanded_queries.extend(expansion_lines)
                self.logger.info(f"Generated query expansions: {expanded_queries}")
            return expanded_queries
        except Exception as e:
            self.logger.error(f"Error in query expansion: {str(e)}")
            return [query]  # Return original query if expansion fails

    def keyword_search(self, query, k=5):
        """Perform keyword-based search using BM25"""
        if not self.has_bm25:
            return []
            
        try:
            results = self.bm25_retriever.get_relevant_documents(query, k=k)
            return results
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            return []

    def merge_results(self, vector_results, keyword_results, mmr_results):
        """Merge results from different retrieval methods, removing duplicates"""
        all_docs = vector_results.copy()
        seen_ids = set(doc.metadata.get('chunk_id', '') for doc in all_docs)
        
        # Add keyword results
        for doc in keyword_results:
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id and chunk_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(chunk_id)
        
        # Add MMR results
        for doc in mmr_results:
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id and chunk_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(chunk_id)
                
        return all_docs

    def rerank_results(self, query, results, top_k=5):
        """Rerank results using cross-encoder"""
        if not self.has_reranker or not results:
            return results[:top_k]
            
        try:
            pairs = [[query, doc.page_content] for doc in results]
            scores = self.reranker.predict(pairs)
            
            # Combine with original documents and sort by score
            scored_docs = list(zip(scores, results))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k results
            reranked_docs = [doc for _, doc in scored_docs[:top_k]]
            return reranked_docs
        except Exception as e:
            self.logger.error(f"Error in reranking: {str(e)}")
            return results[:top_k]  # Return original results if reranking fails

    def retrieve(self, query, metadata_filters=None):
        """Enhanced retrieval combining multiple methods and reranking"""
        # Log the query for debugging
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Query expansion
            expanded_queries = self.expand_query(query)
            
            # Initialize empty results
            all_vector_docs = []
            all_keyword_docs = []
            all_mmr_docs = []
            
            # Process each expanded query
            for expanded_query in expanded_queries:
                # Vector similarity search
                self.logger.info(f"Performing similarity search for '{expanded_query}' with k={self.similarity_k}")
                similarity_docs = self.vector_store.similarity_search(
                    expanded_query, 
                    k=self.similarity_k,
                    filter=metadata_filters
                )
                all_vector_docs.extend(similarity_docs)
                
                # Keyword search using BM25
                keyword_docs = self.keyword_search(expanded_query, k=self.similarity_k)
                all_keyword_docs.extend(keyword_docs)
                
                # MMR search for diversity
                self.logger.info(f"Performing MMR search for '{expanded_query}' with k={self.mmr_k}")
                mmr_docs = self.vector_store.max_marginal_relevance_search(
                    expanded_query,
                    k=self.mmr_k,
                    filter=metadata_filters
                )
                all_mmr_docs.extend(mmr_docs)
            
            # Log retrieved documents for debugging
            self.logger.info(f"Vector search retrieved: {[doc.metadata.get('source') for doc in all_vector_docs]}")
            self.logger.info(f"Keyword search retrieved: {[doc.metadata.get('source') for doc in all_keyword_docs if 'source' in doc.metadata]}")
            self.logger.info(f"MMR search retrieved: {[doc.metadata.get('source') for doc in all_mmr_docs]}")
            
            # Merge results
            combined_docs = self.merge_results(all_vector_docs, all_keyword_docs, all_mmr_docs)
            
            # Rerank combined results
            reranked_docs = self.rerank_results(query, combined_docs, top_k=10)
            
            # Format context string
            context = "\n\n---\n\n".join([
                f"[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" 
                for doc in reranked_docs
            ])
            
            # Print summary for debugging
            print(f"Retrieved {len(reranked_docs)} documents for query: {query}")
            print(f"Sources: {[doc.metadata.get('source') for doc in reranked_docs]}")
            
            self.logger.info(f"Retrieved {len(reranked_docs)} documents for query: {query}")
            
            return {
                "documents": reranked_docs,
                "context": context
            }
        except Exception as e:
            self.logger.error(f"Error in retrieval: {str(e)}")
            print(f"Retrieval error: {str(e)}")
            # Return empty results on error
            return {
                "documents": [],
                "context": f"Error retrieving documents: {str(e)}"
            }