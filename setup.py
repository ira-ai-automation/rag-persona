#!/usr/bin/env python3
"""
Setup script for Local RAG Assistant.
This script handles the complete setup process including virtual environment creation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def check_system_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check available memory (rough estimate)
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split()[1])
                mem_gb = mem_bytes / (1024**3)
                print(f"Available memory: {mem_gb:.1f} GB")
                if mem_gb < 8:
                    print("Warning: Less than 8GB RAM detected. Performance may be limited.")
        elif platform.system() == "Linux":
            result = subprocess.run(["free", "-b"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                mem_line = lines[1].split()
                mem_bytes = int(mem_line[1])
                mem_gb = mem_bytes / (1024**3)
                print(f"Available memory: {mem_gb:.1f} GB")
                if mem_gb < 8:
                    print("Warning: Less than 8GB RAM detected. Performance may be limited.")
    except Exception:
        print("Could not check memory, proceeding anyway...")
    
    return True


def create_virtual_environment():
    """Create Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    success = run_command(f"{sys.executable} -m venv venv")
    
    if success:
        print("Virtual environment created ✓")
        return True
    else:
        print("Failed to create virtual environment")
        return False


def get_pip_command():
    """Get the appropriate pip command for the virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"


def install_dependencies():
    """Install Python dependencies."""
    pip_cmd = get_pip_command()
    
    print("Upgrading pip...")
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print("Warning: Could not upgrade pip")
    
    print("Installing dependencies...")
    success = run_command(f"{pip_cmd} install -r requirements.txt")
    
    if success:
        print("Dependencies installed ✓")
        return True
    else:
        print("Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "models",
        "data", 
        "index",
        "logs",
        "licenses"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Directories created ✓")
    return True


def download_sample_model():
    """Optionally download a sample model."""
    print("\nWould you like to download a sample model? (This may take a while)")
    print("Options:")
    print("1. Skip download (you can download later with 'make download-model')")
    print("2. Download lightweight model (~2.4GB)")
    print("3. Download recommended model (~4.4GB)")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("Skipping model download")
            return True
        elif choice == "2":
            print("Downloading lightweight model...")
            url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
            return run_command(f"curl -L '{url}' -o models/phi-3-mini-4k-instruct-q4.gguf")
        elif choice == "3":
            print("Downloading recommended model...")
            url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            return run_command(f"curl -L '{url}' -o models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        else:
            print("Invalid choice, skipping download")
            return True
            
    except KeyboardInterrupt:
        print("\nSkipping model download")
        return True


def setup_license():
    """Setup licensing system."""
    print("\nSetting up licensing system...")
    
    python_cmd = "venv/bin/python" if platform.system() != "Windows" else "venv\\Scripts\\python"
    
    setup_script = f"""
import sys
sys.path.append('.')
from src.licensing.generator import LicenseGenerator
from src.utils.config import load_config

try:
    config = load_config()
    generator = LicenseGenerator(config)
    result = generator.setup_licensing()
    print("License setup complete!")
    print(f"Demo license: {{result['demo_license_path']}}")
    print(f"Dev license: {{result['dev_license_path']}}")
except Exception as e:
    print(f"License setup failed: {{e}}")
"""
    
    script_path = Path("temp_setup_license.py")
    script_path.write_text(setup_script)
    
    try:
        success = run_command(f"{python_cmd} temp_setup_license.py")
        if success:
            print("Licensing system setup ✓")
    finally:
        if script_path.exists():
            script_path.unlink()
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 LOCAL RAG ASSISTANT SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print()
    print("1. Add documents to scan:")
    print("   • Manual: Place files in data/ folder")
    print("   • Home: Run 'make ingest-home'")
    print("   • System: Run 'sudo make ingest-system'")
    print()
    print("2. Build knowledge base:")
    print("   make ingest          # Manual mode")
    print("   make ingest-home     # Home directory")
    print("   sudo make ingest-system  # System-wide")
    print()
    print("3. Run the assistant:")
    print("   make run             # CLI interface")
    print("   make web             # Web interface")
    print("   make api             # API server")
    print()
    print("4. Other useful commands:")
    print("   make help            # Show all commands")
    print("   make download-model  # Download models")
    print("   make stats           # Show statistics")
    print()
    print("Documentation: README.md")
    print("Configuration: config/settings.yaml")
    print("="*60)


def main():
    """Main setup function."""
    print("Local RAG Assistant Setup")
    print("=" * 30)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check requirements
    if not check_system_requirements():
        return False
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Setting up licensing", setup_license),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"Setup failed at: {step_name}")
            return False
    
    # Optional model download
    download_sample_model()
    
    # Success
    print_next_steps()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed with error: {e}")
        sys.exit(1)
