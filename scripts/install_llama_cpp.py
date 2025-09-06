#!/usr/bin/env python3
"""
Script to install llama-cpp-python with proper platform detection.
This handles the complex compilation process separately.
"""

import platform
import subprocess
import sys
from pathlib import Path


def get_platform_wheel_url():
    """Get platform-specific wheel URL for llama-cpp-python."""
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    # Map architecture names
    if arch in ['x86_64', 'amd64']:
        arch = 'x86_64'
    elif arch in ['arm64', 'aarch64']:
        arch = 'arm64'
    elif arch in ['armv7l']:
        arch = 'armv7l'
    
    if system == 'darwin':  # macOS
        if arch == 'arm64':
            return "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp311-cp311-macosx_11_0_arm64.whl"
        else:
            return "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp311-cp311-macosx_10_9_x86_64.whl"
    elif system == 'linux':
        return "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl"
    elif system == 'windows':
        return "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp311-cp311-win_amd64.whl"
    
    return None


def install_llama_cpp():
    """Install llama-cpp-python using the best method for the platform."""
    print("Installing llama-cpp-python...")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Try pre-built wheel first
    wheel_url = get_platform_wheel_url()
    
    if wheel_url:
        print(f"Trying pre-built wheel: {wheel_url}")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', wheel_url
            ], check=True, capture_output=True, text=True)
            print("✓ Successfully installed pre-built wheel")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Pre-built wheel failed: {e}")
            print("Falling back to compilation...")
    
    # Try extra index URL method
    print("Trying extra index URL...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'llama-cpp-python',
            '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cpu'
        ], check=True, capture_output=True, text=True)
        print("✓ Successfully installed from extra index")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Extra index URL failed: {e}")
    
    # Last resort: compile from source
    print("Compiling from source (this may take 10-15 minutes)...")
    print("You can cancel with Ctrl+C and install manually later.")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'llama-cpp-python>=0.2.0'
        ], check=True)
        print("✓ Successfully compiled from source")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user")
        return False


def main():
    """Main function."""
    print("llama-cpp-python Installation Script")
    print("=" * 40)
    
    # Check if already installed
    try:
        import llama_cpp
        print("llama-cpp-python is already installed!")
        print(f"Version: {llama_cpp.__version__}")
        return True
    except ImportError:
        pass
    
    success = install_llama_cpp()
    
    if success:
        print("\n✓ Installation complete!")
        print("You can now use the Local RAG Assistant with local LLM models.")
    else:
        print("\n✗ Installation failed!")
        print("\nManual installation options:")
        print("1. For CPU-only: pip install llama-cpp-python")
        print("2. For GPU (CUDA): CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
        print("3. For macOS Metal: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
        
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
