# enhanced_retriever.py
import logging
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from agent.gemma_model import GemmaModel
from langchain_core.documents import Document

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
        self.initialize_bm25()

    def initialize_bm25(self):
        """Initialize or reinitialize the BM25 retriever"""
        try:
            # Check if we have a valid vector store
            if not self.vector_store or not self.vector_store._collection:
                self.logger.warning("No vector store available for BM25 initialization")
                self.has_bm25 = False
                return
                
            all_docs = self.vector_store.get()
            if all_docs and 'documents' in all_docs and len(all_docs['documents']) > 0:
                # Create documents with metadata
                documents = []
                for doc_text, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                    doc = Document(page_content=doc_text, metadata=metadata)
                    documents.append(doc)
                
                self.logger.info(f"Initializing BM25 with {len(documents)} documents")
                self.logger.info(f"Document sources: {[doc.metadata.get('source', 'Unknown') for doc in documents]}")
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.has_bm25 = True
                self.logger.info("Successfully initialized BM25 retriever")
            else:
                self.logger.warning("No documents found for BM25 retriever")
                self.has_bm25 = False
        except Exception as e:
            self.logger.error(f"Error initializing BM25 retriever: {str(e)}")
            self.has_bm25 = False

    def expand_query(self, query):
        """Expand the query using the LLM to improve retrieval"""
        self.logger.info(f"Expanding query: {query}")
        try:
            # Check if the query contains proper nouns
            query_terms = query.split()
            has_proper_nouns = any(term.istitle() for term in query_terms)
            
            if has_proper_nouns:
                # Special expansion for queries with proper nouns (names, entities)
                proper_nouns = [term for term in query_terms if term.istitle()]
                self.logger.info(f"Query contains proper nouns: {proper_nouns}")
                
                prompt = f"""Given the user query: "{query}" which contains the name/entity: {', '.join(proper_nouns)}
                Generate 3-5 alternative ways to ask about this specific name/entity.
                Focus on very explicit mentions of the name(s).
                Return only the expanded queries, one per line, no numbering or other text."""
            else:
                # Standard expansion for general queries
                prompt = f"""Given the user query: "{query}"
                Generate 3-5 alternative ways to phrase this query to improve document retrieval.
                Focus on expanding with relevant professional and technical terms.
                Return only the expanded queries, one per line, no numbering or other text."""
            
            response = self.llm.invoke(prompt)
            
            # Always include the original query as the first one
            expanded_queries = [query]
            
            if response and hasattr(response, 'content'):
                expansion_lines = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
                expanded_queries.extend(expansion_lines)
                
                # For queries with proper nouns, add explicit variations
                if has_proper_nouns:
                    for proper_noun in proper_nouns:
                        expanded_queries.append(f"information about {proper_noun}")
                        expanded_queries.append(f"tell me about {proper_noun}")
                        expanded_queries.append(f"{proper_noun} details")
                        expanded_queries.append(f"who is {proper_noun}")
                
                self.logger.info(f"Generated query expansions: {expanded_queries}")
            
            return expanded_queries
        except Exception as e:
            self.logger.error(f"Error in query expansion: {str(e)}")
            return [query]  # Return original query if expansion fails

    def keyword_search(self, query, k=5):
        """Perform keyword-based search using BM25"""
        if not self.has_bm25:
            self.logger.warning("BM25 retriever not initialized, attempting to initialize...")
            self.initialize_bm25()
            
        if not self.has_bm25:
            self.logger.warning("BM25 retriever still not initialized after attempt")
            return []
            
        try:
            self.logger.info(f"Performing keyword search for: {query}")
            results = self.bm25_retriever.get_relevant_documents(query, k=k)
            self.logger.info(f"Found {len(results)} results from keyword search")
            for doc in results:
                self.logger.info(f"Keyword search result: {doc.page_content[:200]}...")
            return results
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            return []

    def merge_results(self, vector_results, keyword_results, mmr_results):
        """Merge results from different retrieval methods, removing duplicates"""
        # Use a dictionary to store unique documents by their content
        unique_docs = {}
        
        # Process vector results first (they tend to be most relevant)
        for doc in vector_results:
            content = doc.page_content.strip()
            if content not in unique_docs:
                unique_docs[content] = doc
        
        # Add keyword results
        for doc in keyword_results:
            content = doc.page_content.strip()
            if content not in unique_docs:
                unique_docs[content] = doc
        
        # Add MMR results
        for doc in mmr_results:
            content = doc.page_content.strip()
            if content not in unique_docs:
                unique_docs[content] = doc
                
        # Convert back to list
        return list(unique_docs.values())

    def rerank_results(self, query, results, top_k=5):
        """Rerank results using cross-encoder"""
        if not results:
            return []
            
        try:
            # Check for proper nouns and rare terms in the query
            query_terms = query.split()
            proper_nouns = [term for term in query_terms if term.istitle()]
            rare_terms = [term for term in query_terms if len(term) > 3]
            
            # Combine proper nouns and rare terms
            important_terms = set(proper_nouns + rare_terms)
            self.logger.info(f"Important query terms identified: {important_terms}")
            
            # If we have a reranker, use it
            if self.has_reranker:
                pairs = [[query, doc.page_content] for doc in results]
                scores = self.reranker.predict(pairs)
                
                # Combine with original documents and sort by score
                scored_docs = list(zip(scores, results))
                
                # Boost documents containing important terms
                if important_terms:
                    boosted_scored_docs = []
                    for score, doc in scored_docs:
                        content = doc.page_content.lower()
                        boost = 0
                        
                        # Check for exact matches of important terms
                        for term in important_terms:
                            if term.lower() in content:
                                # Significant boost for exact matches of proper nouns and rare terms
                                boost += 2.0
                                self.logger.info(f"Boosting document with term '{term}': {doc.metadata.get('source')}")
                        
                        # Apply the boost
                        boosted_scored_docs.append((score + boost, doc))
                    
                    # Sort by boosted score
                    boosted_scored_docs.sort(key=lambda x: x[0], reverse=True)
                    scored_docs = boosted_scored_docs
                
                # Return top_k results
                reranked_docs = [doc for _, doc in scored_docs[:top_k]]
                self.logger.info(f"Reranked results: {[doc.metadata.get('source') for doc in reranked_docs]}")
                return reranked_docs
            else:
                # If we don't have a reranker, implement a simple term-matching algorithm
                scored_docs = []
                for doc in results:
                    content = doc.page_content.lower()
                    score = 0
                    
                    # Core scoring: count query term matches
                    for term in query_terms:
                        if term.lower() in content:
                            score += 0.5
                    
                    # Extra boost for important terms
                    for term in important_terms:
                        if term.lower() in content:
                            score += 2.0
                            self.logger.info(f"Boosting document with term '{term}': {doc.metadata.get('source')}")
                    
                    scored_docs.append((score, doc))
                
                # Sort by score
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                simple_reranked = [doc for _, doc in scored_docs[:top_k]]
                self.logger.info(f"Simple reranked results: {[doc.metadata.get('source') for doc in simple_reranked]}")
                return simple_reranked
        except Exception as e:
            self.logger.error(f"Error in reranking: {str(e)}")
            # Fall back to simple term matching if reranking fails
            try:
                # Sort documents by how many query terms they contain
                query_terms = set(query.lower().split())
                scored_docs = [(sum(1 for term in query_terms if term in doc.page_content.lower()), doc) 
                               for doc in results]
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_docs[:top_k]]
            except:
                # If all else fails, just return the original results
                return results[:top_k]

    def retrieve(self, query, metadata_filters=None):
        """Enhanced retrieval combining multiple methods and reranking"""
        # Log the query for debugging
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Check if we have any documents
            if not self.vector_store or not self.vector_store._collection or self.vector_store._collection.count() == 0:
                self.logger.warning("No documents available in the vector store")
                return {
                    "documents": [],
                    "context": "No documents available in the knowledge base. Please upload some documents first."
                }
            
            # Extract proper nouns for entity-aware retrieval
            query_terms = query.split()
            proper_nouns = [term for term in query_terms if term.istitle()]
            is_entity_query = bool(proper_nouns)
            
            if is_entity_query:
                self.logger.info(f"Entity-aware retrieval activated for: {proper_nouns}")
                
                # Direct search for proper nouns in all documents (brute force approach)
                # This ensures we find names even if they appear only once in a document
                try:
                    all_docs = self.vector_store.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        direct_matches = []
                        
                        for i, (doc_text, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                            # Check for exact name matches in document content
                            for name in proper_nouns:
                                # Try different case variations
                                name_variations = [name, name.lower(), name.upper(), name.title()]
                                
                                for name_var in name_variations:
                                    if name_var in doc_text:
                                        self.logger.info(f"Direct name match found: '{name}' in document {metadata.get('source')}")
                                        # Create a document with the content
                                        direct_match = Document(
                                            page_content=doc_text,
                                            metadata=metadata
                                        )
                                        direct_matches.append(direct_match)
                                        break  # Break the inner loop once a match is found
                        
                        # If we found direct matches, use them directly or add to our results later
                        if direct_matches:
                            self.logger.info(f"Found {len(direct_matches)} direct name matches for {proper_nouns}")
                            # We'll use these matches later in the pipeline
                except Exception as e:
                    self.logger.error(f"Error during direct name search: {str(e)}")
                    direct_matches = []
            else:
                direct_matches = []
            
            # Get all available document sources first
            all_sources = set()
            
            # Get initial list of sources for reference
            try:
                source_counts = self.vector_store.list_documents()
                all_sources = set(source_counts.keys())
                self.logger.info(f"Available document sources: {all_sources}")
            except Exception as e:
                self.logger.error(f"Error getting document sources: {str(e)}")
            
            # Use a safety flag to ensure we don't end up with empty filters
            use_metadata_filters = False
            
            # First, try to find which documents contain relevant information
            # For entity queries, use a direct approach first
            if is_entity_query:
                # For entity queries, increase k substantially to ensure we catch all mentions
                initial_search = self.vector_store.similarity_search(query, k=25)
            else:
                initial_search = self.vector_store.similarity_search(query, k=15)
                
            relevant_sources = set()
            
            if initial_search:
                # Get the sources of all relevant documents
                for doc in initial_search:
                    source = doc.metadata.get('source')
                    if source:
                        relevant_sources.add(source)
                        self.logger.info(f"Found relevant document source: {source}")
                        self.logger.info(f"Document content preview: {doc.page_content[:200]}...")
                
                # For entity queries, force search across all documents unless we found exact matches
                if is_entity_query:
                    # Check if any of the initial results contain the entity by name
                    entity_found = False
                    for doc in initial_search:
                        for entity in proper_nouns:
                            # Check multiple case variations
                            if entity in doc.page_content or entity.lower() in doc.page_content.lower():
                                entity_found = True
                                self.logger.info(f"Found exact entity match for '{entity}' in document: {doc.metadata.get('source')}")
                                break
                    
                    # Also check if we found direct matches earlier
                    if direct_matches:
                        entity_found = True
                        # Add the sources from direct matches
                        for doc in direct_matches:
                            source = doc.metadata.get('source')
                            if source:
                                relevant_sources.add(source)
                                self.logger.info(f"Adding source from direct match: {source}")
                    
                    # If we didn't find an exact match, search all documents
                    if not entity_found:
                        self.logger.info("No exact entity matches found in initial search, searching all documents")
                        relevant_sources = all_sources.copy() if all_sources else set()
                        use_metadata_filters = bool(relevant_sources)
                    else:
                        # We found some matches, use those documents
                        use_metadata_filters = bool(relevant_sources)
                else:
                    # Safety check - if we found documents but no sources, there may be a metadata issue
                    if not relevant_sources and len(initial_search) > 0:
                        self.logger.warning("Found documents but no sources in metadata, using all documents")
                        use_metadata_filters = False
                    # If no relevant sources found from similarity search or fewer than all sources
                    elif not relevant_sources or len(relevant_sources) < len(all_sources):
                        # For queries with rare/specific terms, it's better to search across all documents
                        # This helps with names, technical terms, or rare entities like "Rehan"
                        if len(query.split()) <= 3 or any(term.istitle() for term in query.split()):
                            self.logger.info("Short query or contains proper nouns, searching across all documents")
                            relevant_sources = all_sources.copy() if all_sources else set()
                            # Only use metadata filters if we actually have sources
                            use_metadata_filters = bool(relevant_sources)
                        else:
                            # If we have some relevant sources, use them
                            use_metadata_filters = bool(relevant_sources)
                    else:
                        # We have all relevant sources, use filtering
                        use_metadata_filters = True
                
                # Update metadata filters to only search in relevant documents
                if use_metadata_filters and relevant_sources:
                    metadata_filters = {"source": {"$in": list(relevant_sources)}}
                    self.logger.info(f"Updated metadata filters to search in sources: {relevant_sources}")
                else:
                    metadata_filters = None
                    self.logger.info("Not using metadata filters for this query")
            else:
                # If initial search returned nothing, don't use filters
                metadata_filters = None
                self.logger.info("Initial search returned no results, not using metadata filters")
            
            # Query expansion
            expanded_queries = self.expand_query(query)
            
            # For entity queries, ensure we include specific entity-focused searches
            if is_entity_query:
                # Add direct entity search queries if not already included
                for entity in proper_nouns:
                    entity_queries = [
                        entity,  # Just the name itself
                        entity.lower(),  # Lowercase variant
                        entity.upper(),  # Uppercase variant
                        f"information about {entity}",  # About format
                        f"details about {entity}",  # Details format
                        f"who is {entity}",  # Who is format
                        f"{entity}'s",  # Possessive
                        f"{entity}s"  # Possible plural
                    ]
                    for eq in entity_queries:
                        if eq not in expanded_queries:
                            expanded_queries.append(eq)
                
                self.logger.info(f"Added entity-specific queries: {expanded_queries}")
            
            # Initialize empty results
            all_vector_docs = []
            all_keyword_docs = []
            all_mmr_docs = []
            
            # First add any direct name matches we found earlier
            if direct_matches:
                all_vector_docs.extend(direct_matches)
                self.logger.info(f"Added {len(direct_matches)} direct name matches to results")
            
            # Process each expanded query
            for expanded_query in expanded_queries:
                # Vector similarity search with increased k for rare terms
                k_value = self.similarity_k * 4 if is_entity_query else self.similarity_k * 2
                self.logger.info(f"Performing similarity search for '{expanded_query}' with k={k_value}")
                similarity_docs = self.vector_store.similarity_search(
                    expanded_query, 
                    k=k_value,
                    filter=metadata_filters
                )
                
                # Log the content of retrieved documents
                self.logger.info(f"Found {len(similarity_docs)} similar documents")
                for doc in similarity_docs:
                    self.logger.info(f"Content: {doc.page_content}")
                    self.logger.info(f"Metadata: {doc.metadata}")
                
                all_vector_docs.extend(similarity_docs)
                
                # Keyword search using BM25 with exact matching
                if self.has_bm25:
                    k_value = self.similarity_k * 4 if is_entity_query else self.similarity_k * 2
                    keyword_docs = self.keyword_search(expanded_query, k=k_value)
                    # Filter keyword results by relevant sources if needed
                    if use_metadata_filters and relevant_sources:
                        keyword_docs = [doc for doc in keyword_docs 
                                      if doc.metadata.get('source') in relevant_sources]
                        self.logger.info(f"Filtered keyword results to sources: {relevant_sources}")
                    all_keyword_docs.extend(keyword_docs)
                
                # MMR search for diversity with increased k
                mmr_k = self.mmr_k * 4 if is_entity_query else self.mmr_k * 2
                self.logger.info(f"Performing MMR search for '{expanded_query}' with k={mmr_k}")
                mmr_docs = self.vector_store.max_marginal_relevance_search(
                    expanded_query,
                    k=mmr_k,
                    filter=metadata_filters
                )
                all_mmr_docs.extend(mmr_docs)
            
            # Check if we found any documents at all
            if not all_vector_docs and not all_keyword_docs and not all_mmr_docs:
                self.logger.warning("No documents retrieved with filters, trying without filters")
                # Try again without any filters as a fallback
                similarity_docs = self.vector_store.similarity_search(
                    query, 
                    k=self.similarity_k * 8 if is_entity_query else self.similarity_k * 4
                )
                all_vector_docs.extend(similarity_docs)
                
                # If we have BM25, try that without filters too
                if self.has_bm25:
                    keyword_docs = self.keyword_search(query, k=self.similarity_k * 8 if is_entity_query else self.similarity_k * 4)
                    all_keyword_docs.extend(keyword_docs)
            
            # Log retrieved documents for debugging
            self.logger.info(f"Vector search retrieved: {[doc.metadata.get('source') for doc in all_vector_docs]}")
            self.logger.info(f"Keyword search retrieved: {[doc.metadata.get('source') for doc in all_keyword_docs if 'source' in doc.metadata]}")
            self.logger.info(f"MMR search retrieved: {[doc.metadata.get('source') for doc in all_mmr_docs]}")
            
            # Merge results with deduplication
            combined_docs = self.merge_results(all_vector_docs, all_keyword_docs, all_mmr_docs)
            
            # If we still have no documents, return a clear error
            if not combined_docs:
                self.logger.error("No documents retrieved even after fallback")
                return {
                    "documents": [],
                    "context": "No relevant documents found for your query. Please try a different question or upload more documents."
                }
            
            # For entity queries, apply an even stronger pre-filter before reranking
            if is_entity_query and proper_nouns:
                # Pre-filter to prioritize documents with exact entity matches
                entity_docs = []
                other_docs = []
                
                for doc in combined_docs:
                    has_entity = False
                    for entity in proper_nouns:
                        # Check for the entity in the content (case-insensitive)
                        if (entity.lower() in doc.page_content.lower() or 
                            entity in doc.page_content or 
                            entity.upper() in doc.page_content):
                            has_entity = True
                            break
                    
                    if has_entity:
                        entity_docs.append(doc)
                    else:
                        other_docs.append(doc)
                
                # If we found documents with exact entity matches, prioritize them
                if entity_docs:
                    self.logger.info(f"Found {len(entity_docs)} documents with exact entity matches for {proper_nouns}")
                    # Put entity docs first, followed by others (up to our limit)
                    filtered_docs = entity_docs + other_docs
                else:
                    filtered_docs = combined_docs
                    
                # Use the filtered docs for reranking
                combined_docs = filtered_docs
            
            # Rerank combined results - prioritize name matches even more for entity queries
            top_k = 10
            if is_entity_query:
                # For entity queries, we'll rerank with a stronger emphasis on exact matches
                reranked_docs = self.rerank_results(query, combined_docs, top_k=top_k)
                
                # Double-check that we have name matches in our final results
                # If not, force include direct matches at the top of results
                if direct_matches:
                    has_name_match = False
                    for doc in reranked_docs[:3]:  # Check top 3 results
                        for name in proper_nouns:
                            if name.lower() in doc.page_content.lower():
                                has_name_match = True
                                break
                        if has_name_match:
                            break
                    
                    # If no name match in top results, insert direct matches at the top
                    if not has_name_match:
                        self.logger.info("No name matches in top reranked results, inserting direct matches")
                        # Combine direct matches with reranked (keeping total at top_k)
                        combined_reranked = direct_matches + [doc for doc in reranked_docs if doc not in direct_matches]
                        reranked_docs = combined_reranked[:top_k]
            else:
                # Standard reranking for non-entity queries
                reranked_docs = self.rerank_results(query, combined_docs, top_k=top_k)
            
            # Format context string with original filenames and line numbers
            context_parts = []
            for doc in reranked_docs:
                # Get the original filename from metadata
                filename = doc.metadata.get('original_filename', doc.metadata.get('source', 'Unknown'))
                
                # Calculate approximate line numbers based on chunk position
                content_lines = doc.page_content.split('\n')
                line_count = len(content_lines)
                
                # Format the document reference
                doc_ref = f"[Document: {filename} (Lines {1}-{line_count})]"
                context_parts.append(f"{doc_ref}\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Print summary for debugging
            print(f"Retrieved {len(reranked_docs)} unique documents for query: {query}")
            print(f"Sources: {[doc.metadata.get('original_filename', doc.metadata.get('source')) for doc in reranked_docs]}")
            
            self.logger.info(f"Retrieved {len(reranked_docs)} unique documents for query: {query}")
            
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