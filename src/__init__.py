"""
Local RAG Assistant - Standalone Software

A fully offline Retrieval-Augmented Generation system for personalized AI assistance.
"""

__version__ = "1.0.0"
__author__ = "Local RAG Assistant Team"

from .utils.config import load_config
from .core.pipeline import RAGPipeline

__all__ = ["load_config", "RAGPipeline"]
