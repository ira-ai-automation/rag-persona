"""User interface modules for Local RAG Assistant."""

from .cli import main as cli_main

# Optional imports for web and API interfaces
try:
    from .web import main as web_main
except ImportError:
    web_main = None

try:
    from .api import app as api_app
except ImportError:
    api_app = None

__all__ = ["cli_main", "web_main", "api_app"]
