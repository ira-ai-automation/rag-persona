#!/usr/bin/env python3
"""Script to set environment variables that prevent segfaults."""

import os
import sys

def set_safe_environment():
    """Set environment variables to prevent segfaults and crashes."""
    # Disable Metal acceleration on macOS
    os.environ['GGML_METAL'] = '0'
    os.environ['GGML_METAL_ENABLE'] = '0'
    
    # Disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Disable OpenMP to prevent threading issues
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Disable MKL threading
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Disable multiprocessing for sentence-transformers
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Force single thread for everything
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    print("Environment variables set for maximum stability:")
    for key in ['GGML_METAL', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 
                'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM']:
        print(f"  {key}={os.environ.get(key, 'NOT_SET')}")

if __name__ == '__main__':
    set_safe_environment()
    
    # If arguments provided, execute the command
    if len(sys.argv) > 1:
        import subprocess
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)
