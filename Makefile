# Local RAG Assistant - Makefile for Automation

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
CONFIG_DIR := config
SRC_DIR := src
MODELS_DIR := models
DATA_DIR := data
INDEX_DIR := index

# Default target
.PHONY: help
help:
	@echo "Local RAG Assistant - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make easy-setup     - Automated setup with interactive prompts"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install dependencies only"
	@echo "  make clean          - Clean temporary files and caches"
	@echo "  make clean-all      - Clean everything including models and data"
	@echo ""
	@echo "Model Management:"
	@echo "  make download-model - Download recommended model"
	@echo "  make list-models    - List available models"
	@echo "  make model-info     - Show current model information"
	@echo ""
	@echo "Data & Index:"
	@echo "  make ingest         - Process and index documents (manual mode)"
	@echo "  make ingest-home    - Index documents from user home directory"
	@echo "  make ingest-system  - Index documents from entire system (requires sudo)"
	@echo "  make ingest-custom  - Index documents from custom directories"
	@echo "  make reindex        - Clear and rebuild the entire index"
	@echo "  make index-stats    - Show index statistics"
	@echo ""
	@echo "Running the Assistant:"
	@echo "  make run            - Start CLI interface"
	@echo "  make web            - Start Streamlit web interface"
	@echo "  make api            - Start FastAPI server"
	@echo ""
	@echo "Licensing:"
	@echo "  make setup-license  - Generate RSA keys and sample licenses"
	@echo "  make demo-license   - Generate demo license"
	@echo "  make dev-license    - Generate development license"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code with black and isort"
	@echo "  make validate       - Validate setup and configuration"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs           - Show recent logs"
	@echo "  make status         - Show system status"
	@echo "  make backup         - Backup index and configuration"

# Setup and Installation
.PHONY: setup
setup: $(VENV)/bin/activate
	@echo "Setting up Local RAG Assistant..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p $(MODELS_DIR) $(DATA_DIR) $(INDEX_DIR) logs licenses
	@echo "Setup complete! Next steps:"
	@echo "1. Place documents in $(DATA_DIR)/"
	@echo "2. Download a model: make download-model"
	@echo "3. Index documents: make ingest"
	@echo "4. Start assistant: make run"

.PHONY: easy-setup
easy-setup:
	@echo "Running automated setup script..."
	python3 setup.py

$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Model Management
.PHONY: download-model
download-model:
	@echo "Downloading recommended model (Mistral-7B-Instruct)..."
	@if [ ! -f "$(MODELS_DIR)/mistral-7b-instruct-v0.2.Q4_K_M.gguf" ]; then \
		echo "Downloading from HuggingFace..."; \
		curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
			-o "$(MODELS_DIR)/mistral-7b-instruct-v0.2.Q4_K_M.gguf"; \
		echo "Model downloaded successfully!"; \
		echo "Update config/settings.yaml to use: ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"; \
	else \
		echo "Model already exists at $(MODELS_DIR)/mistral-7b-instruct-v0.2.Q4_K_M.gguf"; \
	fi

.PHONY: download-model-lightweight
download-model-lightweight:
	@echo "Downloading lightweight model (Phi-3-Mini)..."
	@if [ ! -f "$(MODELS_DIR)/phi-3-mini-4k-instruct-q4.gguf" ]; then \
		echo "Downloading from HuggingFace..."; \
		curl -L "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
			-o "$(MODELS_DIR)/phi-3-mini-4k-instruct-q4.gguf"; \
		echo "Lightweight model downloaded successfully!"; \
	else \
		echo "Model already exists"; \
	fi

.PHONY: list-models
list-models:
	@echo "Available models in $(MODELS_DIR):"
	@ls -lh $(MODELS_DIR)/ 2>/dev/null || echo "No models found. Run 'make download-model' first."

.PHONY: model-info
model-info:
	$(VENV_PYTHON) -c "from $(SRC_DIR).utils.config import load_model_config; import json; print(json.dumps(load_model_config(), indent=2))"

# Data Processing and Indexing
.PHONY: ingest
ingest:
	@echo "Processing and indexing documents (manual mode)..."
	$(VENV_PYTHON) -m $(SRC_DIR).data.indexer --scan-mode manual
	@echo "Indexing complete!"

.PHONY: ingest-home
ingest-home:
	@echo "Processing and indexing documents from home directory..."
	$(VENV_PYTHON) -m $(SRC_DIR).data.indexer --scan-mode home
	@echo "Home directory indexing complete!"

.PHONY: ingest-system
ingest-system:
	@echo "Processing and indexing documents from entire system..."
	@echo "This requires sudo privileges and will scan the entire filesystem."
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	sudo $(VENV_PYTHON) -m $(SRC_DIR).data.indexer --scan-mode system
	@echo "System-wide indexing complete!"

.PHONY: ingest-custom
ingest-custom:
	@echo "Processing and indexing documents from custom directories..."
	@echo "Configure custom directories in config/settings.yaml first."
	$(VENV_PYTHON) -m $(SRC_DIR).data.indexer --scan-mode custom
	@echo "Custom directory indexing complete!"

.PHONY: reindex
reindex:
	@echo "Clearing existing index..."
	rm -rf $(INDEX_DIR)/*
	@echo "Rebuilding index..."
	$(MAKE) ingest

.PHONY: index-stats
index-stats:
	$(VENV_PYTHON) -c "from $(SRC_DIR).core.pipeline import RAGPipeline; from $(SRC_DIR).utils.config import load_config; pipeline = RAGPipeline(load_config()); print('Index Statistics:'); import json; print(json.dumps(pipeline.get_stats(), indent=2))"

# Running the Assistant
.PHONY: run
run:
	@echo "Starting Local RAG Assistant (CLI)..."
	$(VENV_PYTHON) fix_environment.py $(VENV_PYTHON) -m $(SRC_DIR).interfaces.cli

.PHONY: web
web:
	@echo "Starting Streamlit web interface..."
	$(VENV_PYTHON) -m streamlit run $(SRC_DIR)/interfaces/web.py

.PHONY: api
api:
	@echo "Starting FastAPI server..."
	$(VENV_PYTHON) -m uvicorn $(SRC_DIR).interfaces.api:app --reload

# Licensing
.PHONY: setup-license
setup-license:
	@echo "Setting up licensing system..."
	$(VENV_PYTHON) -c "from $(SRC_DIR).licensing.generator import LicenseGenerator; from $(SRC_DIR).utils.config import load_config; gen = LicenseGenerator(load_config()); result = gen.setup_licensing(); print('License setup complete!'); print(f'Demo license: {result[\"demo_license_path\"]}'); print(f'Dev license: {result[\"dev_license_path\"]}')"

.PHONY: demo-license
demo-license:
	$(VENV_PYTHON) -c "from $(SRC_DIR).licensing.generator import LicenseGenerator; from $(SRC_DIR).utils.config import load_config; gen = LicenseGenerator(load_config()); token = gen.create_demo_license(); path = gen.save_license(token, 'demo_license.txt'); print(f'Demo license saved to: {path}')"

.PHONY: dev-license
dev-license:
	$(VENV_PYTHON) -c "from $(SRC_DIR).licensing.generator import LicenseGenerator; from $(SRC_DIR).utils.config import load_config; gen = LicenseGenerator(load_config()); token = gen.create_development_license(); path = gen.save_license(token, 'dev_license.txt'); print(f'Development license saved to: {path}')"

# Development and Testing
.PHONY: test
test:
	$(VENV_PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(VENV_PYTHON) -m flake8 $(SRC_DIR)/
	$(VENV_PYTHON) -m isort --check-only $(SRC_DIR)/
	$(VENV_PYTHON) -m black --check $(SRC_DIR)/

.PHONY: format
format:
	$(VENV_PYTHON) -m isort $(SRC_DIR)/
	$(VENV_PYTHON) -m black $(SRC_DIR)/

.PHONY: validate
validate:
	@echo "Validating setup..."
	$(VENV_PYTHON) -c "from $(SRC_DIR).core.pipeline import RAGPipeline; from $(SRC_DIR).utils.config import load_config; pipeline = RAGPipeline(load_config()); results = pipeline.validate_setup(); print('Validation Results:'); import json; print(json.dumps(results, indent=2)); exit(0 if results['can_initialize'] else 1)"

# Utilities
.PHONY: logs
logs:
	@echo "Recent logs:"
	@tail -50 logs/*.log 2>/dev/null || echo "No log files found"

.PHONY: status
status:
	@echo "=== Local RAG Assistant Status ==="
	@echo "Virtual Environment: $(shell test -d $(VENV) && echo 'EXISTS' || echo 'NOT FOUND')"
	@echo "Models Directory: $(shell ls -1 $(MODELS_DIR) 2>/dev/null | wc -l) files"
	@echo "Data Directory: $(shell ls -1 $(DATA_DIR) 2>/dev/null | wc -l) files"
	@echo "Index Directory: $(shell ls -1 $(INDEX_DIR) 2>/dev/null | wc -l) files"
	@echo "Configuration: $(shell test -f $(CONFIG_DIR)/settings.yaml && echo 'EXISTS' || echo 'NOT FOUND')"
	@echo "Last Index Update: $(shell stat -f '%Sm' $(INDEX_DIR)/faiss.index 2>/dev/null || echo 'Never')"

.PHONY: backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf "backups/rag-assistant-backup-$(shell date +%Y%m%d_%H%M%S).tar.gz" \
		$(CONFIG_DIR)/ $(INDEX_DIR)/ licenses/ --exclude='*.log'
	@echo "Backup created in backups/"

# Cleanup
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf *.egg-info
	@echo "Cleanup complete"

.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all data (models, index, data)..."
	rm -rf $(MODELS_DIR)/* $(INDEX_DIR)/* $(DATA_DIR)/* logs/* licenses/*
	@echo "All data cleaned"

# Development shortcuts
.PHONY: dev
dev: setup download-model-lightweight setup-license
	@echo "Development environment ready!"
	@echo "Add some documents to $(DATA_DIR)/ and run 'make ingest'"

.PHONY: quick-start
quick-start: setup
	@echo "Quick start setup..."
	$(MAKE) download-model-lightweight
	@echo "Add your documents to $(DATA_DIR)/ directory"
	@echo "Then run: make ingest && make run"
