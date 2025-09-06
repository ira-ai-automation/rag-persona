"""Document embedding functionality for Local RAG Assistant."""

import logging
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

from ..utils.config import Configuration
from ..utils.logging import get_logger


class DocumentEmbedder:
    """Handles document embedding using sentence transformers."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the document embedder.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def load_model(self) -> None:
        """Load the embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.config.embedding.model_name}")
            
            device = self.config.embedding.device
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
            
            self.model = SentenceTransformer(
                self.config.embedding.model_name,
                device=device
            )
            
            self.logger.info(f"Embedding model loaded successfully on {device}")
            self.logger.info(f"Model dimension: {self.get_embedding_dimension()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension.
            
        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.model.get_sentence_embedding_dimension()
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for processing. If None, uses config value.
            show_progress: Whether to show progress bar.
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
            
        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If texts is empty.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            raise ValueError("Empty text list provided")
        
        if batch_size is None:
            batch_size = self.config.embedding.batch_size
        
        self.logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        
        try:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if len(non_empty_texts) != len(texts):
                self.logger.warning(f"Filtered out {len(texts) - len(non_empty_texts)} empty texts")
            
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.embedding.normalize_embeddings
            )
            
            # Handle case where some texts were empty
            if len(non_empty_texts) != len(texts):
                # Create full array and fill in embeddings for non-empty texts
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
                non_empty_idx = 0
                for i, text in enumerate(texts):
                    if text.strip():
                        full_embeddings[i] = embeddings[non_empty_idx]
                        non_empty_idx += 1
                embeddings = full_embeddings
            
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding array with shape (embedding_dim,).
        """
        # Check cache first if caching is enabled
        if self.config.performance.cache_embeddings:
            text_hash = str(hash(text))
            if text_hash in self._embedding_cache:
                return self._embedding_cache[text_hash]
        
        embeddings = self.embed_texts([text], show_progress=False)
        embedding = embeddings[0]
        
        # Cache the result if caching is enabled
        if self.config.performance.cache_embeddings:
            if len(self._embedding_cache) >= self.config.performance.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            self._embedding_cache[text_hash] = embedding
        
        return embedding
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (alias for embed_text for clarity).
        
        Args:
            query: Query string to embed.
            
        Returns:
            Query embedding array.
        """
        return self.embed_text(query)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        # Ensure embeddings are normalized
        if self.config.embedding.normalize_embeddings:
            # If embeddings are already normalized, dot product = cosine similarity
            return float(np.dot(embedding1, embedding2))
        else:
            # Calculate cosine similarity manually
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.config.embedding.model_name,
            "device": str(self.model.device),
            "dimension": self.get_embedding_dimension(),
            "max_length": self.config.embedding.max_length,
            "normalize_embeddings": self.config.embedding.normalize_embeddings,
            "cache_enabled": self.config.performance.get("cache_embeddings", False),
            "cache_size": len(self._embedding_cache) if self.config.performance.get("cache_embeddings", False) else 0
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, '_embedding_cache'):
            self._embedding_cache.clear()
