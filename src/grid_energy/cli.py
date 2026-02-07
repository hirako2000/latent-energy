import typer
import torch
from typing import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from grid_energy.data.ingestion import fetch_bronze_data
from grid_energy.data.processor import process_silver_data
from grid_energy.data.synthesizer import synthesize_gold
import sys

app = typer.Typer(help="Grid Energy: Structural Integrity for Logic.")
console = Console()

@app.command()
def ingest(
    tier: Annotated[str, typer.Argument(help="The Medallion tier to process")] = "bronze"
):
    """
    Execute either Medallion Pipeline stages.
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
    """Train the EBM to recognize logical equilibrium (low energy)."""
    from grid_energy.train import train_ebm
    
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
    steps: int = typer.Option(100, help="Number of kinetic vibration steps"),
):
    """Execute Kinetic Resolution. Picks random if no ID is provided."""
    import torch
    import random
    from grid_energy.data.loader import CroissantDataset
    from grid_energy.core.models import NonogramCNN
    from grid_energy.core.energy import NonogramEnergy
    from grid_energy.core.solver import KineticSolver
    from grid_energy.utils.config import settings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CroissantDataset()
    
    if not puzzle_id or puzzle_id.strip() == "":
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        puzzle_id = sample["id"]
        console.print(f"[dim]No ID provided. Selected random puzzle: [bold]{puzzle_id}[/][/dim]")
    else:
        try:
            target_idx = ds.ids.index(puzzle_id)
            sample = ds[target_idx]
        except ValueError:
            console.print(f"[red]Error: Puzzle {puzzle_id} not found in Gold layer.[/red]")
            return

    model = NonogramCNN(grid_size=12).to(device)
    weights_path = settings.ROOT_DIR / "data/models/ebm_weights.pt"
    
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        console.print("[dim]Loaded pre-trained kinetic weights.[/dim]")
    else:
        console.print("[yellow]Warning: No weights found. Running with raw physics.[/yellow]")

    model.eval()
    energy_fn = NonogramEnergy(model=model)
    solver = KineticSolver(energy_fn, n_steps=steps)

    console.print(f"[bold cyan]Resolving {puzzle_id}...[/bold cyan]")
    
    # start from noise or a blank slate
    init_state = torch.randn((1, 1, 12, 12)).to(device)
    hints = sample["hints"].unsqueeze(0).to(device)
    
    # solver "vibrates" the grid into equilibrium
    # prime the energy function with context before solving
    energy_fn.set_context(hints)
    resolved = solver.resolve(init_state, hints)
    
    # continuous energy state back to binary grid
    final_grid = (torch.sigmoid(resolved) > 0.5).int().squeeze().cpu().numpy()
    
    console.print("\n[bold green]Equilibrium Reached:[/bold green]")
    for row in final_grid:
        row_str = "".join(["█ " if cell == 1 else "· " for cell in row])
        console.print(f"  {row_str}")

@app.command()
def diagnose(puzzle_id: str):
    """Compare the Energy levels of truth vs. model output."""
    import torch
    from grid_energy.data.loader import CroissantDataset
    from grid_energy.core.models import NonogramCNN
    from grid_energy.core.energy import NonogramEnergy
    from grid_energy.utils.config import settings

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
    """List available puzzle IDs in the Gold dataset."""
    from grid_energy.data.loader import CroissantDataset
    
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