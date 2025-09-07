"""LLM generation functionality for Local RAG Assistant."""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from ..utils.config import Configuration
from ..utils.logging import get_logger
from .retriever import RetrievalResult


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    text: str
    tokens_generated: int
    generation_time: float
    sources: List[str]
    metadata: Dict[str, Any]


class LLMGenerator:
    """Handles text generation using local LLM via llama.cpp."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the LLM generator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.model: Optional[Llama] = None
        
        if Llama is None:
            self.logger.warning("llama-cpp-python not available - generation features disabled")
    
    def __del__(self):
        """Cleanup resources when the object is deleted."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up model resources to prevent memory leaks."""
        if self.model is not None:
            try:
                self.logger.debug("Cleaning up LLM model resources")
                # Force garbage collection of the model
                del self.model
                self.model = None
            except Exception as e:
                self.logger.warning(f"Error during model cleanup: {e}")
    
    def load_model(self) -> None:
        """Load the LLM model."""
        if Llama is None:
            self.logger.warning("Cannot load model: llama-cpp-python not installed")
            return
            
        model_path = Path(self.config.llm.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.logger.info(f"Loading LLM model from {model_path}")
            
            # Determine number of threads - limit to avoid segfaults on macOS
            n_threads = self.config.llm.threads
            if n_threads <= 0:
                import os
                n_threads = min(os.cpu_count() or 4, 4)  # Cap at 4 threads to prevent issues
            else:
                n_threads = min(n_threads, 4)  # Cap at 4 threads maximum
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.llm.context_length,
                n_threads=n_threads,
                verbose=False,
                seed=-1,  # Use random seed
                n_batch=512,  # Reduce batch size to prevent memory issues
                use_mmap=True,  # Use memory mapping for better memory management
                use_mlock=False,  # Disable memory locking to prevent issues
                n_gpu_layers=0  # Force CPU-only mode to avoid GPU-related segfaults
            )
            
            self.logger.info("LLM model loaded successfully")
            self.logger.info(f"Context length: {self.config.llm.context_length}")
            self.logger.info(f"Threads: {n_threads}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> GenerationResult:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            repeat_penalty: Repetition penalty.
            stop_sequences: List of stop sequences.
            stream: Whether to use streaming generation.
            
        Returns:
            Generation result.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        if max_tokens is None:
            max_tokens = self.config.llm.max_tokens
        if temperature is None:
            temperature = self.config.llm.temperature
        if top_p is None:
            top_p = self.config.llm.top_p
        if top_k is None:
            top_k = self.config.llm.top_k
        if repeat_penalty is None:
            repeat_penalty = self.config.llm.repeat_penalty
        
        try:
            start_time = time.time()
            
            if stream:
                # Streaming generation
                response_text = ""
                for chunk in self._generate_stream(
                    prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop_sequences
                ):
                    response_text += chunk
                
                generation_time = time.time() - start_time
                tokens_generated = len(response_text.split())  # Rough estimate
                
            else:
                # Non-streaming generation
                response = self.model(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences or [],
                    echo=False
                )
                
                generation_time = time.time() - start_time
                response_text = response['choices'][0]['text'].strip()
                tokens_generated = response['usage']['completion_tokens']
            
            self.logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
            
            return GenerationResult(
                text=response_text,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                sources=[],  # Will be populated by RAG pipeline
                metadata={
                    'model_path': self.config.llm.model_path,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'max_tokens': max_tokens,
                    'repeat_penalty': repeat_penalty
                }
            )
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_stream(
        self, 
        prompt: str, 
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop_sequences: Optional[List[str]]
    ) -> Generator[str, None, None]:
        """
        Generate text using streaming.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            repeat_penalty: Repetition penalty.
            stop_sequences: List of stop sequences.
            
        Yields:
            Generated text chunks.
        """
        stream = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences or [],
            stream=True,
            echo=False
        )
        
        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    yield delta['content']
                elif 'text' in delta:
                    yield delta['text']
    
    def generate_with_context(
        self, 
        query: str, 
        context_documents: List[RetrievalResult],
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> GenerationResult:
        """
        Generate text with retrieved context documents.
        
        Args:
            query: User query.
            context_documents: Retrieved context documents.
            system_prompt: System prompt override.
            **generation_kwargs: Additional generation parameters.
            
        Returns:
            Generation result with sources.
        """
        # Build context from retrieved documents
        context_text = self._build_context(context_documents)
        
        # Build full prompt
        prompt = self._build_prompt(query, context_text, system_prompt)
        
        # Generate response
        result = self.generate(prompt, **generation_kwargs)
        
        # Add source information
        if self.config.rag.include_sources:
            result.sources = [doc.title for doc in context_documents]
        
        return result
    
    def _build_context(self, documents: List[RetrievalResult]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: Retrieved documents.
            
        Returns:
            Context string.
        """
        if not documents:
            return ""
        
        context_parts = []
        total_length = 0
        max_length = self.config.rag.max_context_length
        
        for doc in documents:
            # Format document
            doc_text = f"[{doc.title}]\n{doc.content}"
            
            # Check if adding this document would exceed max length
            if total_length + len(doc_text) > max_length:
                # Try to fit partial content
                remaining_length = max_length - total_length
                if remaining_length > 100:  # Only add if we have reasonable space
                    truncated_content = doc.content[:remaining_length-len(doc.title)-10] + "..."
                    doc_text = f"[{doc.title}]\n{truncated_content}"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text) + 2  # +2 for separator
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self, 
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the full prompt for generation.
        
        Args:
            query: User query.
            context: Context from retrieved documents.
            system_prompt: System prompt override.
            
        Returns:
            Full prompt string.
        """
        if system_prompt is None:
            system_prompt = self.config.prompts['system']
        
        # Use context template from config
        context_template = self.config.prompts.get('context_template', """
Context from documents:
{context}

Question: {question}

Answer based on the context above:
        """).strip()
        
        if context:
            full_prompt = f"{system_prompt}\n\n{context_template.format(context=context, question=query)}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
        
        return full_prompt
    
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
            "model_path": self.config.llm.model_path,
            "context_length": self.config.llm.context_length,
            "max_tokens": self.config.llm.max_tokens,
            "temperature": self.config.llm.temperature,
            "top_p": self.config.llm.top_p,
            "top_k": self.config.llm.top_k,
            "repeat_penalty": self.config.llm.repeat_penalty,
            "threads": self.config.llm.threads
        }
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate.
            
        Returns:
            Estimated token count.
        """
        # Simple estimation: ~4 characters per token on average
        return len(text) // 4
    
    def can_fit_context(self, context: str, query: str, max_tokens: int = 512) -> bool:
        """
        Check if context and query can fit within model's context window.
        
        Args:
            context: Context text.
            query: Query text.
            max_tokens: Maximum tokens for generation.
            
        Returns:
            True if context fits, False otherwise.
        """
        # Estimate tokens for context, query, system prompt, and generation
        context_tokens = self.estimate_token_count(context)
        query_tokens = self.estimate_token_count(query)
        system_tokens = self.estimate_token_count(self.config.prompts['system'])
        
        total_input_tokens = context_tokens + query_tokens + system_tokens + 50  # Buffer
        total_tokens = total_input_tokens + max_tokens
        
        return total_tokens <= self.config.llm.context_length
