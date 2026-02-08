import hashlib
import itertools
import random
import sys
import time
from typing import Annotated

import numpy as np
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from grid_energy.core.energy import NonogramEnergy
from grid_energy.core.models import NonogramCNN
from grid_energy.core.solver import KineticSolver
from grid_energy.data.ingestion import fetch_bronze_data
from grid_energy.data.loader import CroissantDataset
from grid_energy.data.processor import process_silver_data
from grid_energy.data.synthesizer import synthesize_gold
from grid_energy.train import train_ebm
from grid_energy.utils.config import settings

app = typer.Typer(help="Grid Energy: Structural Integrity for Logic.")
console = Console()

@app.command()
def ingest(
    tier: Annotated[str, typer.Argument(help="The Medallion tier to process")] = "bronze"
):
    """
    Execute Medallion Pipeline stages.
    """
    tier = tier.lower()
    if tier == "bronze":
        typer.echo("Initiating Bronze Ingestion from Hugging Face...")
        fetch_bronze_data()
        typer.echo("Bronze Ingestion Complete.")
    elif tier == "silver":
        typer.echo("Processing Silver Layer: Converting Parquet to Structured Schema...")
        process_silver_data()
        typer.echo("Silver Layer Complete. 'Complexity Atlas' generated in docs/.")
    elif tier == "gold":
        typer.echo("Processing Gold Layer: Synthesizing Croissant Tensors...")
        synthesize_gold()
        typer.echo("Gold Layer Complete. Tensors saved to data/gold/.")
    else:
        typer.echo(f"Error: Unknown tier '{tier}'. Use bronze, silver, or gold.")

@app.command()
def train(
    epochs: int = typer.Option(20, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    lr: float = typer.Option(1e-4, help="Learning rate")
):
    """Train the EBM to recognize logical equilibrium, that is low energy"""

    console.print("[bold yellow]Starting Kinetic Distillation Training...[/bold yellow]")

    try:
        train_ebm(epochs=epochs, batch_size=batch_size, lr=lr)
        console.print("[bold green]Training Complete. Weights stored in data/models/.[/bold green]")
    except Exception as e:
        console.print("\n[bold red]Training Engine Halted:[/bold red]")
        # prints the specific error concisely, adjust for more stack
        console.print(f"[red]» {e}[/red]")
        sys.exit(1)

@app.command()
def resolve(
    puzzle_id: Annotated[str, typer.Option(help="Specific puzzle ID")] = "",
    steps: int = typer.Option(150, help="Number of kinetic vibration steps"),
    compare: bool = typer.Option(True, help="Compare with true solution for analysis"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    max_retries: int = typer.Option(10, help="Max attempts to find a logical solution"),
):
    """
    Execute Kinetic Resolution
    Checks success procedurally by truly counting pixel blocks against hints.
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CroissantDataset()

    if not puzzle_id or puzzle_id.strip() == "":
        random.seed(seed)
        idx = random.randint(0, len(ds) - 1) # noqa: S311
        sample = ds[idx]
        puzzle_id = sample["id"]
        console.print(f"[dim]No ID provided. Selected: [bold]{puzzle_id}[/][/dim]")
    else:
        try:
            target_idx = ds.ids.index(puzzle_id)
            sample = ds[target_idx]
        except ValueError:
            console.print(f"[red]Error: Puzzle {puzzle_id} not found.[/red]")
            return

    model = NonogramCNN(grid_size=12).to(device)
    weights_path = settings.ROOT_DIR / "data/models/ebm_final.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    energy_fn = NonogramEnergy(model=model)
    solver = KineticSolver(energy_fn, n_steps=steps, deterministic=True)

    hints = sample["hints"].unsqueeze(0).to(device)
    target_grid = sample["target_grid"].unsqueeze(0).to(device)

    row_hints_raw = hints[0, 0].cpu().numpy()
    col_hints_raw = hints[0, 1].cpu().numpy()

    # actual puzzle dimensions from non-zero hints
    active_rows = np.where(row_hints_raw.sum(axis=1) > 0)[0]
    active_cols = np.where(col_hints_raw.sum(axis=1) > 0)[0]
    puzzle_size = max(active_rows[-1], active_cols[-1]) + 1 if len(active_rows) > 0 else 12

    # got cleaned hints, no padding
    clean_row_hints = [[h for h in row if h > 0] for row in row_hints_raw[:puzzle_size]]
    clean_col_hints = [[h for h in col if h > 0] for col in col_hints_raw[:puzzle_size]]

    def get_procedural_score(grid_tensor):
        """Counts blocks in the grid and compares them to hints"""
        binary = (torch.sigmoid(grid_tensor * 6.0) > 0.5).int().squeeze().cpu().numpy()
        active_area = binary[:puzzle_size, :puzzle_size]

        total_errors = 0

        for i, target in enumerate(clean_row_hints):
            row = active_area[i, :]
            blocks = [len(list(g)) for k, g in itertools.groupby(row) if k == 1]
            if blocks != target:
                total_errors += 1

        for j, target in enumerate(clean_col_hints):
            col = active_area[:, j]
            blocks = [len(list(g)) for k, g in itertools.groupby(col) if k == 1]
            if blocks != target:
                total_errors += 1

        return total_errors

    hash_obj = hashlib.md5(puzzle_id.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)

    best_resolved = None
    min_errors = 999
    tries_taken = 0
    energy_fn.set_context(hints)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for attempt in range(max_retries):
        tries_taken += 1
        torch.manual_seed((hash_int + attempt) % (2**32))
        init_state = torch.randn((1, 1, 12, 12)).to(device)

        resolved = solver.resolve(init_state, hints)

        error_count = get_procedural_score(resolved)

        if error_count < min_errors:
            min_errors = error_count
            best_resolved = resolved.clone()

        if error_count == 0:
            break

        if attempt < max_retries - 1:
            console.print(f"  [yellow]Attempt {tries_taken}: Logical Errors in {error_count} lines. Retrying...[/yellow]")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start_time) * 1000

    assert best_resolved is not None # noqa: S101
    final_grid = (torch.sigmoid(best_resolved * 6.0) > 0.5).int().squeeze().cpu().numpy()
    final_display = final_grid[:puzzle_size, :puzzle_size]

    console.print(f"\n[bold green]Procedural Equilibrium Reached ({puzzle_size}x{puzzle_size}):[/bold green]")
    for row in final_display:
        console.print(f"  {''.join(['█ ' if cell == 1 else '· ' for cell in row])}")

    console.print("\n[bold]Final Report:[/bold]")
    console.print(f"  Procedural Status: {'[green]SOLVED[/green]' if min_errors == 0 else f'[red]FAILED ({min_errors} lines wrong)[/red]'}")
    console.print(f"  Tries Taken: [bold cyan]{tries_taken}[/bold cyan]")
    console.print(f"  Total Time: [bold magenta]{total_ms:.2f} ms[/bold magenta]")

    if compare:
        target_np = target_grid.int().squeeze().cpu().numpy()[:puzzle_size, :puzzle_size]
        pixel_acc = (final_display == target_np).mean()
        console.print(f"  Dataset Match: {pixel_acc:.2%}")

    return best_resolved

@app.command()
def diagnose(puzzle_id: str):
    """Compare the Energy levels of truth vs. model output"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CroissantDataset()

    try:
        target_idx = ds.ids.index(puzzle_id)
        sample = ds[target_idx]
    except ValueError:
        console.print(f"[red]Error: {puzzle_id} not found.[/red]")
        return

    model = NonogramCNN(grid_size=12).to(device)
    weights_path = settings.ROOT_DIR / "data/models/ebm_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    energy_fn = NonogramEnergy(model=model)

    true_grid = sample["target_grid"].unsqueeze(0).to(device)
    hints = sample["hints"].unsqueeze(0).to(device)
    noise_grid = torch.randn_like(true_grid)

    energy_fn.set_context(hints)

    with torch.no_grad():
        true_energy = energy_fn(true_grid).item()
        noise_energy = energy_fn(noise_grid).item()

    console.print(Panel(f"Energy Analysis: [bold]{puzzle_id}[/]", style="cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("State", style="dim")
    table.add_column("Energy Score (Lower is Better)", justify="right")

    table.add_row("Ground Truth (from Parquet)", f"{true_energy:.4f}")
    table.add_row("Random Noise", f"{noise_energy:.4f}")

    console.print(table)

    if true_energy < noise_energy:
        console.print("[bold green]✓ Reality Check Passed:[/] The model recognizes the true solution as the lower energy state.")
    else:
        console.print("[bold red]✗ Reality Check Failed:[/] The model is 'confused' and prefers noise over the solution.")

@app.command(name="list-ids")
def list_ids(limit: int = typer.Option(10, help="Number of IDs to show")):
    """List available puzzle IDs off the Gold dataset."""

    try:
        ds = CroissantDataset()
        total = len(ds)
        console.print(f"[bold cyan]Available IDs (Total: {total}):[/bold cyan]")

        for i in range(min(limit, total)):
            raw_id = str(ds.ids[i])
            clean_id = (raw_id[:50] + "...") if len(raw_id) > 50 else raw_id

            console.print(f" - {clean_id}", markup=False)

    except Exception as e:
        console.print(f"[bold red]Error loading Gold Layer:[/bold red] {str(e)}", markup=False)

def main():
    app()

if __name__ == "__main__":
    main()
