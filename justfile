# Help
default:
    just --list

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

# aliases for convenience
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

# Resolve a puzzle (random by default, or use id=XYZ. eg puzzle_0060)
resolve id="":
    uv run grid-energy resolve --puzzle-id="{{id}}" --steps 150

diagnose id:
    uv run grid-energy diagnose "{{id}}"

# List a sample of available puzzle IDs
list-puzzles:
    uv run grid-energy list-ids

# --- Linting & Testing ---

# Run all checks
check: lint analysis test

# Fix all auto-fixable linting and formatting issues
fix:
    uv run ruff check --fix .

# Check linting and formatting without fixing 
lint:
    uv run ruff check .

# Run static type analysis
analysis:
    uv run pyright

# Run the test suite with coverage
test *args:
    uv run pytest {{args}} --benchmark-min-rounds=100 -s

# Open the visual coverage report
coverage:
    open htmlcov/index.html


# Run benchmarks for kinetic resolution
benchmark:
    uv run pytest --benchmark-only

# --- Visualizations ---

# Launch the tension dashboard
viz:
    uv run grid-energy resolve demo-puzzle --viz