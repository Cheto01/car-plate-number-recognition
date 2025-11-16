"""Command-line interface for ALPR System."""
import click

from src.core.config import get_settings
from src.core.logging_config import setup_logging
from src.pipeline.alpr_pipeline import ALPRPipeline


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--config", type=str, help="Path to config file")
def cli(debug: bool, config: str):
    """ALPR System - Automatic License Plate Recognition."""
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, help="Output video path")
@click.option("--csv", is_flag=True, help="Save results to CSV")
@click.option("--no-db", is_flag=True, help="Don't save to database")
def process(video_path: str, output: str, csv: bool, no_db: bool):
    """Process a video file."""
    click.echo(f"Processing video: {video_path}")

    pipeline = ALPRPipeline()
    results = pipeline.process_video(
        video_path=video_path,
        output_path=output,
        save_to_db=not no_db,
        save_to_csv=csv,
    )

    click.echo(f"\nProcessing complete!")
    click.echo(f"Frames processed: {results['frames_processed']}")
    click.echo(f"Detections: {results['detections']}")
    click.echo(f"Processing time: {results['processing_time']:.2f}s")


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
def api(host: str, port: int, reload: bool):
    """Start the FastAPI server."""
    import uvicorn

    click.echo(f"Starting API server at http://{host}:{port}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--port", default=8501, help="Dashboard port")
def dashboard(port: int):
    """Start the analytics dashboard."""
    import subprocess

    click.echo(f"Starting dashboard at http://localhost:{port}")

    subprocess.run([
        "streamlit",
        "run",
        "src/analytics/dashboard.py",
        "--server.port",
        str(port),
    ])


@cli.command()
def init_db():
    """Initialize the database."""
    from src.core.database import get_db_manager

    click.echo("Initializing database...")
    db_manager = get_db_manager()
    db_manager.create_tables()
    click.echo("Database initialized successfully!")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
def reset_db():
    """Reset the database (DANGER: deletes all data)."""
    from src.core.database import get_db_manager

    click.echo("Resetting database...")
    db_manager = get_db_manager()
    db_manager.reset_database()
    click.echo("Database reset complete!")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
