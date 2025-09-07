"""Main RAG pipeline for Local RAG Assistant."""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.config import Configuration
from ..utils.logging import get_logger
from .embedder import DocumentEmbedder
from .retriever import VectorRetriever, RetrievalResult
from .generator import LLMGenerator, GenerationResult


@dataclass
class RAGResult:
    """Complete result from RAG pipeline."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """Main RAG pipeline that coordinates all components."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.embedder = DocumentEmbedder(config)
        self.retriever = VectorRetriever(config)
        self.generator = LLMGenerator(config)
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing RAG pipeline...")
            
            # Load embedding model
            self.embedder.load_model()
            
            # Initialize or load vector index
            if not self.retriever.load_index():
                embedding_dim = self.embedder.get_embedding_dimension()
                self.retriever.initialize_index(embedding_dim)
            
            # Load LLM model (optional for indexing)
            try:
                self.generator.load_model()
            except Exception as e:
                self.logger.warning(f"LLM model not available: {e}")
                self.logger.info("Continuing with indexing-only mode")
            
            self._initialized = True
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def query(
        self, 
        query_text: str, 
        k: Optional[int] = None,
        stream: bool = False,
        **generation_kwargs
    ) -> RAGResult:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query_text: User query.
            k: Number of documents to retrieve.
            stream: Whether to use streaming generation.
            **generation_kwargs: Additional generation parameters.
            
        Returns:
            Complete RAG result.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Step 1: Embed the query
            self.logger.debug(f"Processing query: {query_text[:100]}...")
            query_embedding = self.embedder.embed_query(query_text)
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.search(query_embedding, k)
            retrieval_time = time.time() - retrieval_start
            
            self.logger.debug(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
            
            # Step 3: Generate response
            generation_start = time.time()
            
            if retrieved_docs:
                generation_result = self.generator.generate_with_context(
                    query_text, 
                    retrieved_docs,
                    stream=stream,
                    **generation_kwargs
                )
            else:
                # No context found, generate without context
                self.logger.warning("No relevant documents found, generating without context")
                prompt = f"{self.config.prompts['system']}\n\nQuestion: {query_text}\n\nAnswer:"
                generation_result = self.generator.generate(
                    prompt,
                    stream=stream,
                    **generation_kwargs
                )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            # Format sources
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    'title': doc.title,
                    'path': doc.path,
                    'score': doc.score,
                    'chunk_index': doc.chunk_index,
                    'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                })
            
            result = RAGResult(
                query=query_text,
                answer=generation_result.text,
                sources=sources,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                metadata={
                    'documents_retrieved': len(retrieved_docs),
                    'tokens_generated': generation_result.tokens_generated,
                    'generation_metadata': generation_result.metadata,
                    'config_k': k or self.config.rag.retrieval_k,
                    'min_similarity': self.config.rag.min_similarity
                }
            )
            
            self.logger.info(f"Query processed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata.
            
        Returns:
            Number of documents added.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        if not documents:
            return 0
        
        try:
            self.logger.info(f"Adding {len(documents)} documents to knowledge base")
            
            # Extract content for embedding
            contents = [doc['content'] for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedder.embed_texts(contents)
            
            # Add to retriever
            self.retriever.add_documents(embeddings, documents)
            
            self.logger.info(f"Successfully added {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics.
        """
        return {
            'initialized': self._initialized,
            'document_count': self.retriever.get_document_count() if self._initialized else 0,
            'embedder_info': self.embedder.get_model_info() if self._initialized else {},
            'retriever_info': self.retriever.get_index_info() if self._initialized else {},
            'generator_info': self.generator.get_model_info() if self._initialized else {},
            'config': {
                'embedding_model': self.config.embedding.model_name,
                'llm_model': self.config.llm.model_path,
                'chunk_size': self.config.document_processing.chunk_size,
                'retrieval_k': self.config.rag.retrieval_k,
                'context_length': self.config.llm.context_length
            }
        }
    
    def save_index(self) -> None:
        """Save the vector index to disk."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        self.retriever.save_index()
    
    def clear_cache(self) -> None:
        """Clear any caches."""
        if self._initialized:
            self.embedder.clear_cache()
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all components are properly set up.
        
        Returns:
            Dictionary with validation results.
        """
        results = {
            'config_loaded': True,
            'embedding_model_available': False,
            'llm_model_available': False,
            'index_directory_exists': False,
            'can_initialize': False
        }
        
        try:
            # Check if embedding model can be loaded
            test_embedder = DocumentEmbedder(self.config)
            test_embedder.load_model()
            results['embedding_model_available'] = True
        except Exception as e:
            self.logger.warning(f"Embedding model validation failed: {e}")
        
        try:
            # Check if LLM model file exists
            from pathlib import Path
            llm_path = Path(self.config.llm.model_path)
            results['llm_model_available'] = llm_path.exists()
        except Exception as e:
            self.logger.warning(f"LLM model validation failed: {e}")
        
        try:
            # Check index directory
            results['index_directory_exists'] = self.config.paths.index.exists()
        except Exception as e:
            self.logger.warning(f"Index directory validation failed: {e}")
        
        # Overall validation
        results['can_initialize'] = (
            results['embedding_model_available'] and 
            results['llm_model_available'] and 
            results['index_directory_exists']
        )
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._initialized:
            try:
                self.save_index()
            except Exception as e:
                self.logger.warning(f"Failed to save index on exit: {e}")
        
        # Clear any resources
        self.clear_cache()
        self.cleanup()
    
    def cleanup(self):
        """Clean up pipeline resources to prevent memory leaks."""
        try:
            if hasattr(self.generator, 'cleanup'):
                self.generator.cleanup()
            if hasattr(self.embedder, 'cleanup'):
                self.embedder.cleanup()
            if hasattr(self.retriever, 'cleanup'):
                self.retriever.cleanup()
        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")
    
    def __del__(self):
        """Cleanup resources when the object is deleted."""
        self.cleanup()
