"""Command-line interface for Local RAG Assistant."""

import sys
import time
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.markdown import Markdown

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.pipeline import RAGPipeline
from src.utils.config import load_config, ensure_directories
from src.utils.logging import setup_logging
from src.licensing.validator import LicenseValidator


class RAGAssistantCLI:
    """Command-line interface for the RAG Assistant."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.config = None
        self.pipeline = None
        self.license_validator = None
        self.current_license = None
        
    def setup(self) -> bool:
        """Setup the CLI application."""
        try:
            # Load configuration
            self.config = load_config()
            ensure_directories(self.config)
            
            # Setup logging
            log_file = self.config.paths.logs / "rag_assistant.log"
            logging_config = dict(self.config.logging)
            # Rename 'format' to 'log_format' to match function signature
            if 'format' in logging_config:
                logging_config['log_format'] = logging_config.pop('format')
            
            setup_logging(
                log_level=self.config.app.log_level,
                log_file=str(log_file),
                **logging_config
            )
            
            # Initialize license validator
            self.license_validator = LicenseValidator(self.config)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Setup failed: {e}[/red]")
            return False
    
    def check_license(self) -> bool:
        """Check for valid license."""
        if not self.config.licensing.get('enabled', True):
            return True
        
        # Look for license files
        license_files = list(self.config.paths.licenses.glob("*.txt"))
        
        if not license_files:
            self.console.print("[yellow]No license file found. Generating demo license...[/yellow]")
            from src.licensing.generator import LicenseGenerator
            generator = LicenseGenerator(self.config)
            
            if not generator.keys_exist():
                generator.generate_rsa_keys()
            
            demo_license = generator.create_demo_license()
            license_path = generator.save_license(demo_license, "demo_license.txt")
            self.console.print(f"[green]Demo license created: {license_path}[/green]")
            license_files = [license_path]
        
        # Validate the first available license
        for license_file in license_files:
            token = self.license_validator.load_license_from_file(str(license_file))
            if token:
                is_valid, validation_info = self.license_validator.validate_license(token)
                if is_valid:
                    self.current_license = token
                    plan = validation_info['data'].get('plan', 'unknown')
                    remaining = validation_info.get('remaining_queries', 0)
                    self.console.print(f"[green]Valid {plan} license loaded ({remaining} queries remaining)[/green]")
                    return True
                else:
                    self.console.print(f"[red]Invalid license: {validation_info['reason']}[/red]")
        
        self.console.print("[red]No valid license found. Please contact support.[/red]")
        return False
    
    def initialize_pipeline(self) -> bool:
        """Initialize the RAG pipeline."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                
                task = progress.add_task("Initializing RAG pipeline...", total=None)
                
                self.pipeline = RAGPipeline(self.config)
                self.pipeline.initialize()
                
                progress.update(task, description="Pipeline ready!")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Pipeline initialization failed: {e}[/red]")
            return False
    
    def show_welcome(self) -> None:
        """Show welcome message."""
        welcome_text = f"""
# Welcome to Local RAG Assistant

**Version:** {self.config.app.version}
**Model:** {Path(self.config.llm.model_path).name}
**Documents:** {self.pipeline.get_stats()['document_count'] if self.pipeline else 0}

Type your questions and get answers based on your local knowledge base.
Type 'help' for commands, 'quit' to exit.
        """
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="Local RAG Assistant",
            border_style="blue"
        ))
    
    def show_help(self) -> None:
        """Show help information."""
        help_text = """
**Available Commands:**

• `help` - Show this help message
• `stats` - Show system statistics  
• `sources` - Show sources from last query
• `license` - Show license information
• `clear` - Clear screen
• `quit` or `exit` - Exit the application

**Usage:**
Just type your question and press Enter. The assistant will search your
knowledge base and provide an answer with source citations.
        """
        
        self.console.print(Panel(
            Markdown(help_text),
            title="Help",
            border_style="green"
        ))
    
    def show_stats(self) -> None:
        """Show system statistics."""
        if not self.pipeline:
            self.console.print("[red]Pipeline not initialized[/red]")
            return
        
        stats = self.pipeline.get_stats()
        
        table = Table(title="System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Document Count", str(stats['document_count']))
        table.add_row("Embedding Model", stats['config']['embedding_model'])
        table.add_row("LLM Model", Path(stats['config']['llm_model']).name)
        table.add_row("Chunk Size", str(stats['config']['chunk_size']))
        table.add_row("Retrieval K", str(stats['config']['retrieval_k']))
        table.add_row("Context Length", str(stats['config']['context_length']))
        
        if self.current_license:
            usage = self.license_validator.get_license_usage(self.current_license)
            if usage.get('exists'):
                table.add_row("License Plan", usage.get('plan', 'unknown'))
                table.add_row("Total Queries", str(usage.get('total_queries', 0)))
                table.add_row("Daily Queries", str(usage.get('daily_queries', 0)))
        
        self.console.print(table)
    
    def show_license_info(self) -> None:
        """Show license information."""
        if not self.current_license:
            self.console.print("[red]No license loaded[/red]")
            return
        
        is_valid, validation_info = self.license_validator.validate_license(self.current_license)
        usage = self.license_validator.get_license_usage(self.current_license)
        
        license_data = validation_info.get('data', {})
        
        table = Table(title="License Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Plan", license_data.get('plan', 'unknown'))
        table.add_row("User ID", license_data.get('user_id', 'unknown'))
        table.add_row("Valid", "✓" if is_valid else "✗")
        table.add_row("Expires", time.strftime('%Y-%m-%d', time.localtime(license_data.get('expires_at', 0))))
        table.add_row("Max Queries/Day", str(license_data.get('max_queries_per_day', 0)))
        table.add_row("Remaining Today", str(validation_info.get('remaining_queries', 0)))
        
        if usage.get('exists'):
            table.add_row("Total Queries Used", str(usage.get('total_queries', 0)))
            table.add_row("First Used", usage.get('first_used', 'Never'))
            table.add_row("Last Used", usage.get('last_used', 'Never'))
        
        self.console.print(table)
    
    def process_query(self, query: str) -> Optional[dict]:
        """Process a user query."""
        if not self.pipeline:
            self.console.print("[red]Pipeline not initialized[/red]")
            return None
        
        # Record query usage if license is active
        if self.current_license:
            # Pre-validate license
            is_valid, validation_info = self.license_validator.validate_license(self.current_license)
            if not is_valid:
                self.console.print(f"[red]License validation failed: {validation_info['reason']}[/red]")
                return None
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                
                task = progress.add_task("Processing query...", total=None)
                start_time = time.time()
                
                result = self.pipeline.query(query)
                
                processing_time = time.time() - start_time
                progress.update(task, description=f"Query processed in {processing_time:.2f}s")
            
            # Record usage
            if self.current_license:
                self.license_validator.record_query_usage(
                    self.current_license,
                    query_length=len(query),
                    response_length=len(result.answer),
                    processing_time=processing_time
                )
            
            return result
            
        except Exception as e:
            self.console.print(f"[red]Query processing failed: {e}[/red]")
            return None
    
    def display_result(self, result) -> None:
        """Display query result."""
        # Show answer
        self.console.print(Panel(
            Markdown(result.answer),
            title="Answer",
            border_style="green"
        ))
        
        # Show timing info
        timing_text = f"Retrieved in {result.retrieval_time:.2f}s, Generated in {result.generation_time:.2f}s"
        self.console.print(f"[dim]{timing_text}[/dim]")
        
        # Show sources if available
        if result.sources:
            self.console.print("\n[bold cyan]Sources:[/bold cyan]")
            for i, source in enumerate(result.sources, 1):
                self.console.print(f"  {i}. {source['title']} (score: {source['score']:.3f})")
        
        self.last_result = result
    
    def show_sources(self) -> None:
        """Show detailed sources from last query."""
        if not hasattr(self, 'last_result') or not self.last_result:
            self.console.print("[yellow]No previous query results[/yellow]")
            return
        
        if not self.last_result.sources:
            self.console.print("[yellow]No sources available[/yellow]")
            return
        
        for i, source in enumerate(self.last_result.sources, 1):
            panel_content = f"""**File:** {source['path']}
**Score:** {source['score']:.3f}
**Chunk:** {source['chunk_index']}

{source['content_preview']}"""
            
            self.console.print(Panel(
                Markdown(panel_content),
                title=f"Source {i}: {source['title']}",
                border_style="blue"
            ))
    
    def run_interactive(self) -> None:
        """Run the interactive CLI."""
        if not self.setup():
            return
        
        if not self.check_license():
            return
        
        if not self.initialize_pipeline():
            return
        
        self.show_welcome()
        
        try:
            while True:
                query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not query.strip():
                    continue
                
                query_lower = query.lower().strip()
                
                if query_lower in ['quit', 'exit']:
                    if Confirm.ask("Are you sure you want to exit?"):
                        break
                elif query_lower == 'help':
                    self.show_help()
                elif query_lower == 'stats':
                    self.show_stats()
                elif query_lower == 'sources':
                    self.show_sources()
                elif query_lower == 'license':
                    self.show_license_info()
                elif query_lower == 'clear':
                    self.console.clear()
                    self.show_welcome()
                else:
                    result = self.process_query(query)
                    if result:
                        self.display_result(result)
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Goodbye![/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Unexpected error: {e}[/red]")


@click.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--query', '-q', help='Single query mode')
@click.option('--license', '-l', help='Path to license file')
def main(config: Optional[str] = None, query: Optional[str] = None, license: Optional[str] = None):
    """Local RAG Assistant - Command Line Interface."""
    
    cli = RAGAssistantCLI()
    
    if query:
        # Single query mode
        if not cli.setup():
            sys.exit(1)
        
        if not cli.check_license():
            sys.exit(1)
        
        if not cli.initialize_pipeline():
            sys.exit(1)
        
        result = cli.process_query(query)
        if result:
            cli.display_result(result)
        else:
            sys.exit(1)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == '__main__':
    main()
