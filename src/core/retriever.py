"""Vector retrieval functionality for Local RAG Assistant."""

import logging
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.config import Configuration
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    doc_id: int
    title: str
    content: str
    path: str
    score: float
    chunk_index: int


class VectorRetriever:
    """Handles vector-based document retrieval using FAISS."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the vector retriever.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.index: Optional[faiss.Index] = None
        self.metadata_db_path = config.paths.index / "metadata.db"
        
        # Ensure index directory exists
        ensure_directory(config.paths.index)
    
    def initialize_index(self, embedding_dimension: int) -> None:
        """
        Initialize a new FAISS index.
        
        Args:
            embedding_dimension: Dimension of embeddings.
        """
        self.logger.info(f"Initializing new FAISS index with dimension {embedding_dimension}")
        
        # Create appropriate index based on configuration
        if self.config.vector_db.index_type == "IndexFlatIP":
            # Inner Product index (for cosine similarity with normalized vectors)
            base_index = faiss.IndexFlatIP(embedding_dimension)
        elif self.config.vector_db.index_type == "IndexFlatL2":
            # L2 distance index
            base_index = faiss.IndexFlatL2(embedding_dimension)
        elif self.config.vector_db.index_type == "IndexIVFFlat":
            # IVF index for larger datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            base_index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist)
        else:
            # Default to flat IP index
            self.logger.warning(f"Unknown index type {self.config.vector_db.index_type}, using IndexFlatIP")
            base_index = faiss.IndexFlatIP(embedding_dimension)
        
        # Wrap with IDMap to store document IDs
        self.index = faiss.IndexIDMap(base_index)
        
        # Initialize metadata database
        self._initialize_metadata_db()
        
        self.logger.info("FAISS index initialized successfully")
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if index was loaded successfully, False otherwise.
        """
        index_path = self.config.paths.index / "faiss.index"
        
        if not index_path.exists():
            self.logger.info("No existing index found")
            return False
        
        try:
            self.logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            
            # Verify metadata database exists
            if not self.metadata_db_path.exists():
                self.logger.warning("Index file exists but metadata database is missing")
                return False
            
            self.logger.info(f"Index loaded successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        index_path = self.config.paths.index / "faiss.index"
        
        try:
            self.logger.info(f"Saving FAISS index to {index_path}")
            faiss.write_index(self.index, str(index_path))
            self.logger.info("Index saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            embeddings: Document embeddings array with shape (n_docs, embedding_dim).
            metadata: List of metadata dictionaries for each document.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        self.logger.info(f"Adding {len(embeddings)} documents to index")
        
        try:
            # Get document IDs (start from existing count)
            start_id = self.get_document_count()
            doc_ids = np.arange(start_id, start_id + len(embeddings), dtype=np.int64)
            
            # Add to FAISS index
            self.index.add_with_ids(embeddings.astype(np.float32), doc_ids)
            
            # Add metadata to database
            self._add_metadata(doc_ids, metadata)
            
            self.logger.info(f"Successfully added {len(embeddings)} documents")
            
            # Save index periodically
            if self.index.ntotal % self.config.vector_db.save_interval == 0:
                self.save_index()
                
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to retrieve. If None, uses config value.
            
        Returns:
            List of retrieval results.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        if k is None:
            k = self.config.rag.retrieval_k
        
        if self.index.ntotal == 0:
            self.logger.warning("Index is empty")
            return []
        
        try:
            # Ensure query embedding is the right shape and type
            query_embedding = query_embedding.astype(np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search the index
            scores, doc_ids = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Get metadata for results
            results = []
            for score, doc_id in zip(scores[0], doc_ids[0]):
                if doc_id == -1:  # Invalid ID
                    continue
                
                # Apply minimum similarity threshold
                if score < self.config.rag.min_similarity:
                    continue
                
                metadata = self._get_metadata(int(doc_id))
                if metadata:
                    results.append(RetrievalResult(
                        doc_id=int(doc_id),
                        title=metadata['title'],
                        content=metadata['content'],
                        path=metadata['path'],
                        score=float(score),
                        chunk_index=metadata.get('chunk_index', 0)
                    ))
            
            self.logger.debug(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            Number of documents.
        """
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the index.
        
        Returns:
            Dictionary with index information.
        """
        if self.index is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "index_type": type(self.index).__name__,
            "total_documents": self.index.ntotal,
            "dimension": self.index.d,
            "is_trained": self.index.is_trained,
            "metadata_db_exists": self.metadata_db_path.exists()
        }
    
    def _initialize_metadata_db(self) -> None:
        """Initialize the metadata database."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_index INTEGER DEFAULT 0,
                    file_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_path ON documents(path)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Metadata database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metadata database: {e}")
            raise
    
    def _add_metadata(self, doc_ids: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add metadata to the database."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            for doc_id, meta in zip(doc_ids, metadata):
                cursor.execute("""
                    INSERT INTO documents (id, title, content, path, chunk_index, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(doc_id),
                    meta.get('title', ''),
                    meta.get('content', ''),
                    meta.get('path', ''),
                    meta.get('chunk_index', 0),
                    meta.get('file_hash', '')
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to add metadata: {e}")
            raise
    
    def _get_metadata(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a document ID."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, content, path, chunk_index, file_hash, created_at
                FROM documents
                WHERE id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'title': row[0],
                    'content': row[1],
                    'path': row[2],
                    'chunk_index': row[3],
                    'file_hash': row[4],
                    'created_at': row[5]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for doc_id {doc_id}: {e}")
            return None
    
    def remove_documents_by_path(self, file_path: str) -> int:
        """
        Remove documents by file path (useful for updates).
        
        Args:
            file_path: Path of file to remove.
            
        Returns:
            Number of documents removed.
        """
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            # Get document IDs to remove
            cursor.execute("SELECT id FROM documents WHERE path = ?", (file_path,))
            doc_ids = [row[0] for row in cursor.fetchall()]
            
            if not doc_ids:
                conn.close()
                return 0
            
            # Remove from metadata database
            cursor.execute("DELETE FROM documents WHERE path = ?", (file_path,))
            removed_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            # Note: FAISS doesn't support efficient removal, so we'd need to rebuild
            # the index for true removal. For now, we just remove from metadata.
            
            self.logger.info(f"Removed {removed_count} document entries for {file_path}")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove documents for {file_path}: {e}")
            return 0
    
    def cleanup(self) -> None:
        """Clean up retriever resources."""
        try:
            if self.index is not None:
                # Clear FAISS index from memory
                del self.index
                self.index = None
                self.logger.debug("FAISS index cleared from memory")
        except Exception as e:
            self.logger.warning(f"Error during retriever cleanup: {e}")