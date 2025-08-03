"""Command Line Interface for Natural SQL."""

import logging
import sys
from typing import Optional, List
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate

from ..config.settings import get_settings
from ..database.connection import get_db_connection
from ..nlp.query_processor import QueryProcessor
from ..database.models import QueryResult

# Initialize CLI app
app = typer.Typer(
    name="natural-sql",
    help="Convert natural language queries to SQL and execute them.",
    add_completion=False
)

# Rich console for beautiful output
console = Console()

# Global variables
query_processor: Optional[QueryProcessor] = None
db_connection = None
settings = None


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('natural_sql.log'),
            logging.StreamHandler(sys.stdout) if debug else logging.NullHandler()
        ]
    )


def initialize_components() -> bool:
    """Initialize database connection and query processor."""
    global query_processor, db_connection, settings
    
    try:
        settings = get_settings()
        db_connection = get_db_connection()
        
        # Test database connection
        if not db_connection.test_connection():
            console.print("[red]Failed to connect to database. Please check your configuration.[/red]")
            return False
        
        query_processor = QueryProcessor(settings, db_connection)
        console.print("[green]✓ Connected to database successfully[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        return False


def display_query_result(result: QueryResult, output_format: str = "table") -> None:
    """Display query results in the specified format."""
    if not result.is_success:
        console.print(f"[red]Query failed: {result.error}[/red]")
        return
    
    if result.row_count == 0:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    if output_format.lower() == "json":
        import json
        data = []
        for row in result.rows:
            row_dict = dict(zip(result.columns, row))
            data.append(row_dict)
        console.print(json.dumps(data, indent=2, default=str))
        
    elif output_format.lower() == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(result.columns)
        writer.writerows(result.rows)
        console.print(output.getvalue())
        
    else:  # table format (default)
        table = Table(show_header=True, header_style="bold magenta")
        
        # Add columns
        for column in result.columns:
            table.add_column(column)
        
        # Add rows (limit to first 50 for display)
        display_rows = result.rows[:50]
        for row in display_rows:
            # Convert all values to strings for display
            str_row = [str(val) if val is not None else "" for val in row]
            table.add_row(*str_row)
        
        console.print(table)
        
        if result.row_count > 50:
            console.print(f"[yellow]Showing first 50 of {result.row_count} results[/yellow]")
    
    # Show execution time
    console.print(f"[dim]Executed in {result.execution_time:.3f} seconds[/dim]")


@app.command()
def query(
    question: Optional[str] = typer.Argument(None, help="Natural language question"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Explain the query and results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging")
) -> None:
    """Execute a natural language query."""
    setup_logging(debug)
    
    if not initialize_components():
        raise typer.Exit(1)
    
    # Get question from user if not provided
    if not question:
        question = Prompt.ask("Enter your question")
    
    if not question.strip():
        console.print("[red]Question cannot be empty.[/red]")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Processing query...", total=None)
        
        result = query_processor.process_natural_language_query(question)
    
    # Display results
    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[bold]Generated SQL:[/bold] {result.query}")
    console.print()
    
    display_query_result(result, output_format)
    
    # Show explanation if requested
    if explain and result.is_success:
        console.print("\n[bold]Explanation:[/bold]")
        explanation = query_processor.explain_query_results(result.query, result)
        console.print(Panel(explanation, title="Query Explanation", border_style="blue"))


@app.command()
def interactive() -> None:
    """Start interactive mode for continuous querying."""
    setup_logging(False)
    
    if not initialize_components():
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold blue]Natural SQL Interactive Mode[/bold blue]\n"
        "Enter natural language questions to query your database.\n"
        "Commands: /help, /schema, /tables, /history, /modify, /quit",
        border_style="blue"
    ))
    
    query_history = []
    
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]natural-sql>[/bold cyan]", default="")
            
            if not question.strip():
                continue
            
            # Handle special commands
            if question.startswith('/'):
                if question == '/quit' or question == '/exit':
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif question == '/help':
                    show_help()
                elif question == '/schema':
                    show_schema()
                elif question == '/tables':
                    show_tables()
                elif question == '/history':
                    show_history(query_history)
                elif question == '/clear':
                    query_processor.clear_cache()
                    console.print("[green]Cache cleared.[/green]")
                elif question == '/modify':
                    modification_request = Prompt.ask("Enter modification request (INSERT/UPDATE/DELETE)")
                    if modification_request.strip():
                        # Process modification in interactive mode
                        try:
                            result = query_processor.process_natural_language_query(modification_request, admin_mode=True)
                            
                            if not result.is_success:
                                console.print(f"[red]Failed: {result.error}[/red]")
                                continue
                            
                            console.print(f"[bold]Generated SQL:[/bold] {result.query}")
                            
                            # Generate preview if possible
                            preview_query = query_processor.generate_preview_query(result.query)
                            if preview_query:
                                console.print("[bold yellow]Preview:[/bold yellow]")
                                preview_result = db_connection.execute_query(preview_query)
                                if preview_result.is_success:
                                    display_query_result(preview_result)
                            
                            # Ask for confirmation
                            if Confirm.ask("Execute this modification?", default=False):
                                mod_result = db_connection.execute_query(result.query)
                                if mod_result.is_success:
                                    console.print("[green]✓ Modification completed![/green]")
                                else:
                                    console.print(f"[red]✗ Failed: {mod_result.error}[/red]")
                            else:
                                console.print("[yellow]Modification cancelled.[/yellow]")
                                
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/red]")
                else:
                    console.print("[red]Unknown command. Type /help for available commands.[/red]")
                continue
            
            # Process natural language query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                result = query_processor.process_natural_language_query(question)
            
            # Store in history
            query_history.append({
                'question': question,
                'sql': result.query,
                'timestamp': result.timestamp,
                'success': result.is_success
            })
            
            # Display results
            console.print(f"[bold]Generated SQL:[/bold] {result.query}")
            display_query_result(result)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def show_help() -> None:
    """Show help information."""
    help_text = """
[bold]Available Commands:[/bold]
  /help     - Show this help message
  /schema   - Show database schema
  /tables   - List all tables
  /history  - Show query history
  /modify   - Execute data modifications (INSERT/UPDATE/DELETE)
  /clear    - Clear schema cache
  /quit     - Exit interactive mode

[bold]Example Questions:[/bold]
  - "Show me all users"
  - "How many orders were placed last month?"
  - "What are the top 5 products by sales?"
  - "Find customers who have never placed an order"

[bold]Example Modifications (use /modify):[/bold]
  - "Add a new user named John"
  - "Update user age to 25 where name is John"
  - "Delete users older than 65"
    """
    console.print(Panel(help_text, title="Help", border_style="green"))


def show_schema() -> None:
    """Show database schema."""
    try:
        schema = db_connection.get_database_schema()
        
        for table in schema.tables:
            table_info = Table(title=f"Table: {table.name}", show_header=True)
            table_info.add_column("Column", style="cyan")
            table_info.add_column("Type", style="magenta")
            table_info.add_column("Nullable", style="yellow")
            table_info.add_column("Key", style="green")
            
            for column in table.columns:
                table_info.add_row(
                    column['name'],
                    column['type'],
                    "Yes" if column['nullable'] else "No",
                    column['key'] or ""
                )
            
            console.print(table_info)
            console.print()
            
    except Exception as e:
        console.print(f"[red]Failed to get schema: {e}[/red]")


def show_tables() -> None:
    """Show list of all tables."""
    try:
        table_names = db_connection.get_table_names()
        
        table = Table(title="Database Tables", show_header=True)
        table.add_column("Table Name", style="cyan")
        
        for name in table_names:
            table.add_row(name)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to get tables: {e}[/red]")


def show_history(history: List[dict]) -> None:
    """Show query history."""
    if not history:
        console.print("[yellow]No query history available.[/yellow]")
        return
    
    table = Table(title="Query History", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("SQL", style="magenta", max_width=40)
    table.add_column("Status", width=8)
    table.add_column("Time", style="dim")
    
    for i, entry in enumerate(history[-10:], 1):  # Show last 10
        status = "[green]✓[/green]" if entry['success'] else "[red]✗[/red]"
        time_str = entry['timestamp'].strftime("%H:%M:%S")
        
        table.add_row(
            str(i),
            entry['question'][:37] + "..." if len(entry['question']) > 40 else entry['question'],
            entry['sql'][:37] + "..." if len(entry['sql']) > 40 else entry['sql'],
            status,
            time_str
        )
    
    console.print(table)


@app.command()
def test_connection() -> None:
    """Test database connection."""
    setup_logging(False)
    
    console.print("Testing database connection...")
    
    if initialize_components():
        console.print("[green]✓ Database connection successful![/green]")
        
        # Show basic info
        table_names = db_connection.get_table_names()
        console.print(f"Found {len(table_names)} tables: {', '.join(table_names)}")
    else:
        console.print("[red]✗ Database connection failed![/red]")
        raise typer.Exit(1)


@app.command()
def modify(
    question: Optional[str] = typer.Argument(None, help="Natural language modification request"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done without executing"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging")
) -> None:
    """Execute a natural language modification query (INSERT, UPDATE, DELETE) with confirmation."""
    setup_logging(debug)
    
    if not initialize_components():
        raise typer.Exit(1)
    
    # Get question from user if not provided
    if not question:
        question = Prompt.ask("Enter your modification request")
    
    if not question.strip():
        console.print("[red]Modification request cannot be empty.[/red]")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Processing modification request...", total=None)
        
        # Generate the modification query
        result = query_processor.process_natural_language_query(question, admin_mode=True)
    
    if not result.is_success:
        console.print(f"[red]Failed to generate modification query: {result.error}[/red]")
        raise typer.Exit(1)
    
    # Display the generated query
    console.print(f"\n[bold]Modification Request:[/bold] {question}")
    console.print(f"[bold]Generated SQL:[/bold] {result.query}")
    
    # Check if this is a modification query
    query_upper = result.query.upper().strip()
    is_modification = any(query_upper.startswith(keyword) for keyword in ['INSERT', 'UPDATE', 'DELETE'])
    
    if not is_modification:
        console.print("[yellow]This appears to be a SELECT query, not a modification. Use 'natural-sql query' instead.[/yellow]")
        display_query_result(result, output_format)
        return
    
    # Generate preview for DELETE/UPDATE operations
    preview_query = query_processor.generate_preview_query(result.query)
    if preview_query:
        console.print(f"\n[bold yellow]Preview of affected rows:[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Generating preview...", total=None)
            preview_result = db_connection.execute_query(preview_query)
        
        if preview_result.is_success:
            display_query_result(preview_result, output_format)
        else:
            console.print(f"[yellow]Could not generate preview: {preview_result.error}[/yellow]")
    
    # Show dry run information
    if dry_run:
        console.print(f"\n[bold blue]DRY RUN MODE:[/bold blue] The above query would be executed.")
        console.print("[dim]Use without --dry-run to actually execute the modification.[/dim]")
        return
    
    # Ask for confirmation
    console.print(f"\n[bold red]⚠️  WARNING:[/bold red] This will modify your database!")
    
    operation_type = "UNKNOWN"
    if query_upper.startswith('INSERT'):
        operation_type = "INSERT"
    elif query_upper.startswith('UPDATE'):
        operation_type = "UPDATE" 
    elif query_upper.startswith('DELETE'):
        operation_type = "DELETE"
    
    console.print(f"[yellow]Operation type: {operation_type}[/yellow]")
    
    if not Confirm.ask("Do you want to proceed with this modification?", default=False):
        console.print("[yellow]Modification cancelled.[/yellow]")
        return
    
    # Execute the modification
    console.print("\n[bold]Executing modification...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Executing modification...", total=None)
        
        # Execute the original modification query
        modification_result = db_connection.execute_query(result.query)
    
    if modification_result.is_success:
        console.print(f"[green]✓ Modification completed successfully![/green]")
        console.print(f"[dim]Execution time: {modification_result.execution_time:.3f} seconds[/dim]")
        
        # Show affected rows count if available
        if hasattr(modification_result, 'row_count') and modification_result.row_count > 0:
            console.print(f"[dim]Rows affected: {modification_result.row_count}[/dim]")
    else:
        console.print(f"[red]✗ Modification failed: {modification_result.error}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from .. import __version__
    console.print(f"Natural SQL CLI v{__version__}")


if __name__ == "__main__":
    app()