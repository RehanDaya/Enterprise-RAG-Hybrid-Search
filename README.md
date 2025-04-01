Use "streamlit run app/ui/app.py" to launch it

# Enterprise Knowledge RAG System: Technical Overview

This system implements an advanced Retrieval-Augmented Generation (RAG) architecture designed for enterprise knowledge management. Here's a comprehensive breakdown:

## Core Architecture

The system employs a modular pipeline architecture with these key components:

1. **Document Processing Layer**: Handles ingestion and chunking of various document formats (PDF, DOCX, TXT) using LangChain's document loaders and RecursiveCharacterTextSplitter.

2. **Vector Store Layer**: Implements semantic search using BAAI/bge-large-en-v1.5 embeddings and ChromaDB for vector storage.

3. **Advanced Retrieval Layer**: Utilizes a hybrid retrieval approach combining:
   - Dense vector similarity search
   - BM25 sparse retrieval for keyword matching
   - Maximum Marginal Relevance (MMR) for diversity
   - Cross-encoder reranking to prioritize the most relevant results
   - Entity-aware retrieval for proper nouns and rare terms
   - Direct text matching for single-occurrence entities

4. **Query Enhancement**: Employs LLM-based query expansion to generate alternative phrasings and improve retrieval recall.

5. **Response Generation**: Uses a locally-hosted Gemma 7B instruction model via Ollama to generate natural language responses from retrieved contexts.

## Technical Differentiators

- **Multi-strategy Retrieval**: The hybrid approach helps overcome the limitations of purely semantic or keyword-based systems.
- **Entity-Aware Processing**: Specialized handling of proper nouns and rare terms ensures accurate retrieval of specific entities.
- **Local Execution**: Complete privacy with all processing done on-premises.
- **Streamlit Interface**: User-friendly web UI for document uploads and conversational interaction.
- **Citation Support**: Generated responses include source citations for traceability.
- **Evaluation Framework**: Built-in tools for precision/recall measurement.

This implementation aligns with current best practices in enterprise RAG systems, offering a balance of accuracy, privacy, and usability while remaining highly extensible through its modular design.

----------------------------------------------------------------------------

# Enterprise Knowledge RAG System: Comprehensive Technical Analysis

## Document Processing Layer

At its core, the document processing layer serves as the foundation of any RAG system by transforming unstructured content into processable units. When we examine this component more deeply:

The system employs a sophisticated approach to document handling through the `DocumentProcessor` class. This processor doesn't simply extract text; it maintains critical metadata relationships throughout the transformation process. For example, when processing a complex workplace safety manual:

1. **Format-Specific Loading**: Different document formats require specialized extraction techniques. PDFs are processed through `PyPDFLoader`, which handles complex layouts and preserves page boundaries, while other formats use `UnstructuredFileLoader` to maintain document structure.

2. **Intelligent Chunking**: The chunking algorithm employs a recursive approach with a hierarchy of separators (`\n\n`, `\n`, `. `, etc.) to preserve natural document boundaries. This ensures that conceptually related information stays together rather than being artificially separated, which is critical for context preservation.

3. **Metadata Enrichment**: Each chunk is tagged with provenance information (source document, file path) and a unique chunk identifier, creating a traceable lineage from raw document to final retrieval.

This sophisticated processing ensures that the system maintains document coherence while preparing content for vectorization.

## Vector Store Layer

The vector store layer implements the semantic core of the system, enabling meaning-based search beyond simple keyword matching:

1. **High-Dimensional Embedding**: The system utilizes BAAI/bge-large-en-v1.5, an advanced embedding model that projects text into a 1024-dimensional vector space. This particular model excels at capturing fine-grained semantic relationships, especially in technical and enterprise contexts.

2. **Optimized Vector Search**: ChromaDB provides efficient nearest-neighbor search capabilities through HNSW (Hierarchical Navigable Small World) graphs, dramatically reducing search space while maintaining high recall.

3. **Persistence Management**: The implementation includes robust persistence handling to maintain embeddings across sessions, eliminating redundant processing and enabling incremental updates to the knowledge base.

This vector infrastructure enables semantic search that understands conceptual relationships rather than merely matching words, allowing the system to find relevant information even when expressed in different terms.

## Advanced Retrieval Layer

The retrieval layer represents the system's most sophisticated component, implementing a hybrid approach that addresses the limitations of any single retrieval method:

1. **Ensemble Retrieval**: Rather than relying solely on vector similarity, the system integrates multiple retrieval approaches:
   - Dense retrieval captures semantic similarity
   - Sparse retrieval (BM25) excels at keyword matching and rare terms
   - MMR introduces diversity to prevent information redundancy
   - Entity-aware retrieval handles proper nouns and single-occurrence terms

2. **Cross-Encoder Reranking**: The system employs a two-stage retrieval process where an initial broad retrieval is followed by precise reranking. The cross-encoder evaluates query-document pairs directly rather than comparing separate embeddings, dramatically improving precision. This computationally expensive step is only applied to the initially retrieved set, balancing effectiveness with efficiency.

3. **Dynamic Retrieval Parameters**: The system adjusts retrieval parameters based on query characteristics, employing different strategies for different query types:
   - Increased search breadth for entity queries
   - Specialized handling of proper nouns with case-insensitive matching
   - Direct text matching for rare terms and single-occurrence entities
   - Adaptive boosting of documents containing exact entity matches

4. **Entity-Aware Processing**: The system includes specialized handling for proper nouns and rare terms:
   - Direct text matching bypasses embedding-based search for exact name matches
   - Multiple case variations are checked to ensure robust matching
   - Documents containing exact entity matches are prioritized in results
   - Fallback strategies ensure rare terms are found even in single-occurrence contexts

This multilayered approach produces more comprehensive and relevant results than any single method could achieve, with particular attention to the challenges of retrieving specific entities and rare terms.

## Query Enhancement

Query enhancement addresses a fundamental limitation in user-system interaction—the vocabulary mismatch problem:

1. **LLM-Powered Expansion**: The system leverages the Gemma model's language understanding to reformulate the original query into multiple variants that explore different semantic angles.

2. **Technical Term Enrichment**: Particularly valuable in enterprise settings, the expansion process deliberately incorporates domain-specific terminology that might be implied but not explicitly stated in the original query.

3. **Parallel Processing**: The system processes all query variants simultaneously and merges the results, effectively casting a wider retrieval net without sacrificing precision due to the subsequent reranking step.

This approach significantly improves recall while maintaining high precision, addressing the classic recall-precision tradeoff in information retrieval.

## Response Generation

The response generation component transforms retrieved information into coherent, contextually appropriate answers:

1. **Context-Aware Prompting**: Retrieved documents are formulated into a structured prompt that guides the Gemma model to focus on relevant information while providing clear attribution requirements.

2. **Local LLM Deployment**: By using Ollama to host Gemma 7B locally, the system maintains data sovereignty while providing responsive performance. The 7B parameter size represents a sweet spot between capability and computational efficiency.

3. **Source Attribution**: The system embeds source tracking throughout the pipeline, enabling proper citation in the final response and maintaining information provenance.

This approach creates a bridge between raw information retrieval and human-friendly interaction, delivering contextually appropriate responses while maintaining transparency about information sources.

The intricate interplay between these components creates a system that effectively navigates the complex terrain of enterprise knowledge management, balancing precision, recall, performance, and usability in a sophisticated solution that extends well beyond basic RAG implementations.