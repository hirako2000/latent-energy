set shell := ["bash", "-c"]

# Display available commands
help:
    @just --list

# Install dependencies and setup environment
setup:
    uv sync
    uv run pre-commit install

# Run the CLI help directly
cli-help:
    uv run grid-energy --help

# --- Data Engineering ---

# Usage: just inject bronze
inject tier="bronze":
    uv run grid-energy ingest {{tier}}

# Direct aliases for convenience
inject-bronze:
    just inject bronze

inject-silver:
    just inject silver

inject-gold:
    just inject gold

# Process raw data into structured Parquet (Silver)
process-silver:
    uv run grid-energy ingest silver

# Finalize EBM-ready tensors with Croissant metadata (Gold)
prepare-gold:
    uv run grid-energy ingest gold

train:
    uv run grid-energy train --epochs 50

# Resolve a puzzle (random by default, or use id=XYZ)
resolve id="":
    uv run grid-energy resolve --puzzle-id="{{id}}" --steps 150

diagnose id:
    uv run grid-energy diagnose "{{id}}"

# List a sample of available puzzle IDs
list-puzzles:
    uv run grid-energy list-ids

# --- Linting & Testing ---

# Run the full test suite with coverage
test:
    uv run pytest

# Run linting and type checking
lint:
    uv run ruff check .
    uv run pyright

# Run benchmarks for kinetic resolution
benchmark:
    uv run pytest --benchmark-only

# --- Visualizations ---

# Launch the tension dashboard
viz:
    uv run grid-energy resolve demo-puzzle --viz