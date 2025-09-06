"""Helper functions for Local RAG Assistant."""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of file contents.
    
    Args:
        file_path: Path to file.
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256').
        
    Returns:
        Hexadecimal hash string.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If algorithm is not supported.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get hash function
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Read file in chunks to handle large files
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted size string (e.g., '1.5 MB').
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file.
        
    Returns:
        File size in bytes.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string.
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename.
        
    Returns:
        Safe filename.
    """
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Ensure it's not empty and doesn't start with a dot
    if not safe_name or safe_name.startswith('.'):
        safe_name = f"file_{safe_name}"
    
    return safe_name


def timestamp_filename(base_name: str, extension: str = "") -> str:
    """
    Create filename with timestamp.
    
    Args:
        base_name: Base filename.
        extension: File extension (with or without dot).
        
    Returns:
        Timestamped filename.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    
    return f"{base_name}_{timestamp}{extension}"


def is_text_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is likely a text file.
    
    Args:
        file_path: Path to file.
        
    Returns:
        True if file is likely text, False otherwise.
    """
    file_path = Path(file_path)
    
    # Check extension first
    text_extensions = {'.txt', '.md', '.py', '.yaml', '.yml', '.json', '.csv', '.log'}
    if file_path.suffix.lower() in text_extensions:
        return True
    
    # Check file content (sample)
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(512)
            
        # Check for null bytes (common in binary files)
        if b'\x00' in sample:
            return False
            
        # Try to decode as UTF-8
        try:
            sample.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
            
    except (IOError, OSError):
        return False


def create_temp_file(suffix: str = "", prefix: str = "tmp", directory: Optional[str] = None) -> Path:
    """
    Create a temporary file path.
    
    Args:
        suffix: File suffix/extension.
        prefix: File prefix.
        directory: Directory for temp file. If None, uses system temp.
        
    Returns:
        Path to temporary file.
    """
    import tempfile
    
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    os.close(fd)  # Close file descriptor, we just need the path
    
    return Path(path)
