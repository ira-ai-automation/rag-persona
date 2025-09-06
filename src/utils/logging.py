"""Logging configuration for Local RAG Assistant."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    file_rotation: str = "midnight",
    backup_count: int = 7,
    max_file_size: str = "10MB"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs to console only.
        log_format: Log message format. If None, uses default.
        file_rotation: When to rotate log files ('midnight', 'size', or None)
        backup_count: Number of backup files to keep
        max_file_size: Maximum size before rotating (e.g., '10MB')
        
    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("local_rag_assistant")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        if file_rotation == "midnight":
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
        elif file_rotation == "size":
            # Convert size string to bytes
            size_bytes = _parse_size(max_file_size)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=size_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    Parse size string to bytes.
    
    Args:
        size_str: Size string like '10MB', '1GB', etc.
        
    Returns:
        Size in bytes.
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str = "local_rag_assistant") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
