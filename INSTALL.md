# Installation Guide - Local RAG Assistant

## Quick Installation

### Option 1: Automated Setup (Recommended)

The easiest way to get started:

```bash
git clone <repository-url>
cd rag-persona
python3 setup.py
```

The setup script will:
- Check system requirements
- Create virtual environment
- Install all dependencies
- Create necessary directories
- Setup licensing system
- Optionally download models
- Provide next steps

### Option 2: Manual Setup

If you prefer manual control:

```bash
git clone <repository-url>
cd rag-persona
make setup
make download-model
```

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 10GB+ free space
- **OS**: macOS, Linux, or Windows

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 16GB+ for better performance
- **Storage**: 20GB+ for multiple models
- **CPU**: Multi-core processor for parallel processing

## Installation Steps (Manual)

### 1. Clone Repository
```bash
git clone <repository-url>
cd rag-persona
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create Directories
```bash
mkdir -p models data index logs licenses
```

### 5. Download Model
Choose one of these models:

**Lightweight** (2.4GB - good for testing):
```bash
curl -L "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" -o models/phi-3-mini-4k-instruct-q4.gguf
```

**Recommended** (4.4GB - balanced performance):
```bash
curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" -o models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 6. Update Configuration
Edit `config/settings.yaml` to point to your downloaded model:
```yaml
llm:
  model_path: "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

### 7. Setup Licensing
```bash
python -c "from src.licensing.generator import LicenseGenerator; from src.utils.config import load_config; gen = LicenseGenerator(load_config()); gen.setup_licensing()"
```

## Troubleshooting

### Common Issues

#### 1. Python Version Error
```
Error: Python 3.10 or higher is required
```
**Solution**: Install Python 3.10+ from [python.org](https://python.org)

#### 2. Memory Error During Model Loading
```
RuntimeError: Cannot allocate memory
```
**Solutions**:
- Use a smaller model (Phi-3-Mini)
- Close other applications
- Add swap space
- Use a machine with more RAM

#### 3. Permission Error (System Scan)
```
PermissionError: System scan requires sudo privileges
```
**Solution**: Run system scans with sudo:
```bash
sudo make ingest-system
# OR
sudo python scripts/system_scan.py
```

#### 4. CUDA Not Available
```
CUDA requested but not available, falling back to CPU
```
**Solution**: This is normal for CPU-only setups. To use GPU:
- Install CUDA toolkit
- Install GPU-enabled versions of torch and faiss-gpu

#### 5. Model Download Fails
```
curl: error downloading model
```
**Solutions**:
- Check internet connection
- Try downloading manually from HuggingFace
- Use alternative model URLs

### Installation Verification

Test your installation:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Test imports
python -c "from src.core.pipeline import RAGPipeline; print('Core imports: OK')"
python -c "from src.utils.config import load_config; print('Config load: OK')"

# Check model exists
ls -la models/

# Run validation
make validate
```

## Platform-Specific Notes

### macOS
- Use Homebrew for system dependencies: `brew install python3`
- For M1/M2 Macs, ensure you're using native Python (not Rosetta)

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv curl
```

### Linux (CentOS/RHEL)
```bash
sudo yum install python3 python3-pip curl
# OR for newer versions:
sudo dnf install python3 python3-pip curl
```

### Windows
- Install Python from Microsoft Store or python.org
- Use PowerShell or Command Prompt
- Replace `source venv/bin/activate` with `venv\Scripts\activate`

## Docker Installation (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -t local-rag-assistant .

# Run container
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models local-rag-assistant
```

## Next Steps

After successful installation:

1. **Add Documents**: Place files in `data/` or configure scanning
2. **Index Documents**: Run `make ingest` or other scanning modes
3. **Start Assistant**: Run `make run` for CLI or `make web` for web interface

See [README.md](README.md) for usage instructions and examples.

## Getting Help

- Check the [README.md](README.md) for usage instructions
- Review [config/settings.yaml](config/settings.yaml) for configuration options
- Run `make help` for all available commands
- Check logs in `logs/` directory for error details
