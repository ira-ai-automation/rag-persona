"""Configuration management for Local RAG Assistant."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration."""
    name: str
    version: str
    debug: bool
    log_level: str


@dataclass
class PathConfig:
    """Path configuration."""
    models: Path
    data: Path
    index: Path
    logs: Path
    licenses: Path
    config: Path
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str
    device: str
    batch_size: int
    max_length: int
    normalize_embeddings: bool


@dataclass
class DocumentConfig:
    """Document processing configuration."""
    supported_formats: list
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunks_per_doc: int


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    index_type: str
    use_gpu: bool
    nprobe: int
    save_interval: int


@dataclass
class LLMConfig:
    """Language model configuration."""
    model_path: str
    context_length: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    threads: int


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    retrieval_k: int
    rerank: bool
    rerank_k: int
    min_similarity: float
    max_context_length: int
    include_sources: bool


@dataclass
class Configuration:
    """Main configuration class."""
    app: AppConfig
    paths: PathConfig
    embedding: EmbeddingConfig
    document_processing: DocumentConfig
    vector_db: VectorDBConfig
    llm: LLMConfig
    rag: RAGConfig
    prompts: Dict[str, str]
    api: Dict[str, Any]
    web: Dict[str, Any]
    licensing: Dict[str, Any]
    logging: Dict[str, Any]
    performance: Dict[str, Any]
    scanning: Dict[str, Any]


def load_config(config_path: Optional[str] = None) -> Configuration:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Configuration object.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    if config_path is None:
        # Look for config in current directory or parent
        config_path = "config/settings.yaml"
        if not os.path.exists(config_path):
            config_path = "../config/settings.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError("Configuration file not found. Expected at config/settings.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Create configuration objects
    app_config = AppConfig(**config_data['app'])
    path_config = PathConfig(**config_data['paths'])
    embedding_config = EmbeddingConfig(**config_data['embedding'])
    document_config = DocumentConfig(**config_data['document_processing'])
    vector_db_config = VectorDBConfig(**config_data['vector_db'])
    llm_config = LLMConfig(**config_data['llm'])
    rag_config = RAGConfig(**config_data['rag'])
    
    return Configuration(
        app=app_config,
        paths=path_config,
        embedding=embedding_config,
        document_processing=document_config,
        vector_db=vector_db_config,
        llm=llm_config,
        rag=rag_config,
        prompts=config_data['prompts'],
        api=config_data['api'],
        web=config_data['web'],
        licensing=config_data['licensing'],
        logging=config_data['logging'],
        performance=config_data['performance'],
        scanning=config_data.get('scanning', {})
    )


def load_model_config(model_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        model_config_path: Path to model config file. If None, uses default.
        
    Returns:
        Dictionary containing model configurations.
    """
    if model_config_path is None:
        model_config_path = "config/models.yaml"
        if not os.path.exists(model_config_path):
            model_config_path = "../config/models.yaml"
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directories(config: Configuration) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration object.
    """
    directories = [
        config.paths.models,
        config.paths.data,
        config.paths.index,
        config.paths.logs,
        config.paths.licenses
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
