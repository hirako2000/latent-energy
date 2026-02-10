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

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SOLVER_STEPS = 150
DEFAULT_SEED = 42
DEFAULT_MAX_RETRIES = 10
DEFAULT_LIMIT_IDS = 10

# Resolution
SIGMOID_SCALE = 6.0
MAX_ERROR_COUNT_INITIAL = 999
HASH_HEX_LENGTH = 8
HASH_BASE = 16
RANDOM_INT_RANGE = 2**32

CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

CYAN_STYLE = "cyan"
GREEN_STYLE = "green"
RED_STYLE = "red"
YELLOW_STYLE = "yellow"
MAGENTA_STYLE = "magenta"
BOLD_STYLE = "bold"
DIM_STYLE = "dim"

MODEL_WEIGHTS_BEST_PATH = "data/models/ebm_best.pt"
MODEL_WEIGHTS_FINAL_PATH = "data/models/ebm_final.pt"

DEFAULT_PUZZLE_SIZE = 5
DISPLAY_SYMBOL_FILLED = "█ "
DISPLAY_SYMBOL_EMPTY = "· "
PANEL_BORDER_STYLE = "cyan"

SOLVED_STATUS = "SOLVED"
FAILED_STATUS = "FAILED"


def get_device():
    return torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


def load_model(device, weights_path, hint_dim):
    model = NonogramCNN(grid_size=DEFAULT_PUZZLE_SIZE, hint_dim=hint_dim).to(device)
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        except Exception as e:
            console.print(f"[{YELLOW_STYLE}]Warning: Could not load weights: {e}[/{YELLOW_STYLE}]")
            console.print(f"[{YELLOW_STYLE}]Continuing with randomly initialized weights.[/{YELLOW_STYLE}]")
    model.eval()
    return model


def select_puzzle(dataset, puzzle_id, seed):
    if not puzzle_id or puzzle_id.strip() == "":
        random.seed(seed)
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        puzzle_id = sample["id"]
        console.print(f"[{DIM_STYLE}]No ID provided. Selected: [{BOLD_STYLE}]{puzzle_id}[/][/{DIM_STYLE}]")
        return sample, puzzle_id, idx
    else:
        try:
            target_idx = dataset.ids.index(puzzle_id)
            sample = dataset[target_idx]
            return sample, puzzle_id, target_idx
        except ValueError:
            console.print(f"[{RED_STYLE}]Error: Puzzle {puzzle_id} not found.[/{RED_STYLE}]")
            return None, None, None


def calculate_puzzle_size(hints_tensor):
    row_hints_raw = hints_tensor[0, 0].cpu().numpy()
    col_hints_raw = hints_tensor[0, 1].cpu().numpy()
    
    active_rows = np.where(row_hints_raw.sum(axis=1) > 0)[0]
    active_cols = np.where(col_hints_raw.sum(axis=1) > 0)[0]
    
    if len(active_rows) > 0 and len(active_cols) > 0:
        puzzle_size = max(active_rows[-1], active_cols[-1]) + 1
    elif len(active_rows) > 0:
        puzzle_size = active_rows[-1] + 1
    elif len(active_cols) > 0:
        puzzle_size = active_cols[-1] + 1
    else:
        puzzle_size = DEFAULT_PUZZLE_SIZE
    
    return puzzle_size, row_hints_raw, col_hints_raw


def extract_clean_hints(row_hints_raw, col_hints_raw, puzzle_size):
    clean_row_hints = [[h for h in row if h > 0] for row in row_hints_raw[:puzzle_size]]
    clean_col_hints = [[h for h in col if h > 0] for col in col_hints_raw[:puzzle_size]]
    return clean_row_hints, clean_col_hints


def get_procedural_score(grid_tensor, puzzle_size, clean_row_hints, clean_col_hints):
    binary = (torch.sigmoid(grid_tensor * SIGMOID_SCALE) > 0.5).int().squeeze().cpu().numpy()
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


def generate_seed_from_puzzle_id(puzzle_id, attempt):
    hash_obj = hashlib.md5(puzzle_id.encode())
    hash_int = int(hash_obj.hexdigest()[:HASH_HEX_LENGTH], HASH_BASE)
    return (hash_int + attempt) % RANDOM_INT_RANGE

def display_solution(final_grid, puzzle_size):
    final_display = final_grid[:puzzle_size, :puzzle_size]
    
    console.print(f"\n[{BOLD_STYLE}{GREEN_STYLE}]Procedural Equilibrium Reached ({puzzle_size}x{puzzle_size}):[/{BOLD_STYLE}{GREEN_STYLE}]")
    for row in final_display:
        row_str = ''.join([DISPLAY_SYMBOL_FILLED if cell == 1 else DISPLAY_SYMBOL_EMPTY for cell in row])
        console.print(f"  {row_str}")

def resolve_puzzle_with_retries(model, solver, energy_fn, hints, max_retries, puzzle_id,
                               puzzle_size, clean_row_hints, clean_col_hints):
    best_resolved = None
    min_errors = MAX_ERROR_COUNT_INITIAL
    tries_taken = 0
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for attempt in range(max_retries):
        tries_taken += 1
        
        init_state = model.predict_grid(hints)
        energy_fn.set_context(hints)
        resolved = solver.resolve(init_state, hints)
        
        error_count = get_procedural_score(resolved, puzzle_size, clean_row_hints, clean_col_hints)
        
        if error_count < min_errors:
            min_errors = error_count
            best_resolved = resolved.clone()
        
        if error_count == 0:
            break
        
        if attempt < max_retries - 1:
            console.print(f"  [{YELLOW_STYLE}]Attempt {tries_taken}: Logical Errors in {error_count} lines. Retrying...[/{YELLOW_STYLE}]")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start_time) * 1000
    
    return best_resolved, min_errors, tries_taken, total_ms


@app.command()
def ingest(
    tier: Annotated[str, typer.Argument(help="The Medallion tier to process")] = "bronze"
):
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
    epochs: int = typer.Option(DEFAULT_EPOCHS, help="Number of training epochs"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Batch size for training"),
    lr: float = typer.Option(DEFAULT_LEARNING_RATE, help="Learning rate")
):
    console.print(f"[{BOLD_STYLE}{YELLOW_STYLE}]Starting Kinetic Distillation Training...[/{BOLD_STYLE}{YELLOW_STYLE}]")

    try:
        train_ebm(epochs=epochs, batch_size=batch_size, lr=lr)
        console.print(f"[{BOLD_STYLE}{GREEN_STYLE}]Training Complete. Weights stored in data/models/.[/{BOLD_STYLE}{GREEN_STYLE}]")
    except Exception as e:
        console.print(f"\n[{BOLD_STYLE}{RED_STYLE}]Training Engine Halted:[/{BOLD_STYLE}{RED_STYLE}]")
        console.print(f"[{RED_STYLE}]» {e}[/{RED_STYLE}]")
        sys.exit(1)


@app.command()
def resolve(
    puzzle_id: Annotated[str, typer.Option(help="Specific puzzle ID")] = "",
    steps: int = typer.Option(DEFAULT_SOLVER_STEPS, help="Number of kinetic vibration steps"),
    compare: bool = typer.Option(True, help="Compare with true solution for analysis"),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed for reproducibility"),
    max_retries: int = typer.Option(1, help="Max attempts to find a logical solution"),
):
    torch.manual_seed(seed)
    device = get_device()
    ds = CroissantDataset()
    
    result = select_puzzle(ds, puzzle_id, seed)
    if result[0] is None:
        return
    sample, puzzle_id, _ = result
    
    hints = sample["hints"].unsqueeze(0).to(device)
    hint_dim = hints.view(1, -1).size(1)
    
    model = load_model(device, settings.ROOT_DIR / MODEL_WEIGHTS_BEST_PATH, hint_dim)
    energy_fn = NonogramEnergy(model=model)
    solver = KineticSolver(energy_fn, n_steps=steps, deterministic=True)
    
    target_grid = sample["target_grid"].unsqueeze(0).to(device)
    
    puzzle_size, row_hints_raw, col_hints_raw = calculate_puzzle_size(hints)
    clean_row_hints, clean_col_hints = extract_clean_hints(row_hints_raw, col_hints_raw, puzzle_size)
    console.print(f"\n[{BOLD_STYLE}]Puzzle Hints:[/{BOLD_STYLE}]")
    console.print(f"[{DIM_STYLE}]Rows: {[[int(h) for h in row] for row in clean_row_hints]}[/{DIM_STYLE}]")
    console.print(f"[{DIM_STYLE}]Columns: {[[int(h) for h in col] for col in clean_col_hints]}[/{DIM_STYLE}]")

    energy_fn.set_context(hints)
    
    best_resolved, min_errors, tries_taken, total_ms = resolve_puzzle_with_retries(
        model, solver, energy_fn, hints, max_retries, puzzle_id,
        puzzle_size, clean_row_hints, clean_col_hints
    )
    
    if best_resolved is None:
        console.print(f"[{RED_STYLE}]Error: Failed to resolve puzzle[/{RED_STYLE}]")
        return
    
    final_grid = (torch.sigmoid(best_resolved * SIGMOID_SCALE) > 0.5).int().squeeze().cpu().numpy()
    final_display = final_grid[:puzzle_size, :puzzle_size]
    
    console.print(f"\n[{BOLD_STYLE}{GREEN_STYLE}]Procedural Equilibrium Reached ({puzzle_size}x{puzzle_size}):[/{BOLD_STYLE}{GREEN_STYLE}]")
    for row in final_display:
        row_str = ''.join([DISPLAY_SYMBOL_FILLED if cell == 1 else DISPLAY_SYMBOL_EMPTY for cell in row])
        console.print(f"  {row_str}")
    
    console.print(f"\n[{BOLD_STYLE}]Final Report:[/{BOLD_STYLE}]")
    
    if min_errors == 0:
        status_text = f"[{GREEN_STYLE}]{SOLVED_STATUS}[/{GREEN_STYLE}]"
    else:
        status_text = f"[{RED_STYLE}]{FAILED_STATUS} ({min_errors} lines wrong)[/{RED_STYLE}]"
    
    console.print(f"  Procedural Status: {status_text}")
    console.print(f"  Tries Taken: [{BOLD_STYLE}{CYAN_STYLE}]{tries_taken}[/{BOLD_STYLE}{CYAN_STYLE}]")
    console.print(f"  Total Time: [{BOLD_STYLE}{MAGENTA_STYLE}]{total_ms:.2f} ms[/{BOLD_STYLE}{MAGENTA_STYLE}]")
    
    if compare:
        target_np = target_grid.int().squeeze().cpu().numpy()[:puzzle_size, :puzzle_size]
        pixel_acc = (final_display == target_np).mean()
        console.print(f"  Dataset Match: {pixel_acc:.2%}")
        
        console.print(f"\n[{BOLD_STYLE}{CYAN_STYLE}]Comparison: Predicted vs Dataset Solution[/{BOLD_STYLE}{CYAN_STYLE}]")
        console.print(f"[{DIM_STYLE}]Left: Predicted | Right: Dataset[/{DIM_STYLE}]")
        
        for i in range(puzzle_size):
            pred_row = ''.join([DISPLAY_SYMBOL_FILLED if cell == 1 else DISPLAY_SYMBOL_EMPTY
                              for cell in final_display[i]])
            target_row = ''.join([DISPLAY_SYMBOL_FILLED if cell == 1 else DISPLAY_SYMBOL_EMPTY
                                for cell in target_np[i]])
            console.print(f"  {pred_row}   │   {target_row}")
    
    return best_resolved

@app.command(name="list-ids")
def list_ids(limit: int = typer.Option(DEFAULT_LIMIT_IDS, help="Number of IDs to show")):
    try:
        ds = CroissantDataset()
        total = len(ds)
        console.print(f"[{BOLD_STYLE}{CYAN_STYLE}]Available IDs (Total: {total}):[/{BOLD_STYLE}{CYAN_STYLE}]")
        
        for i in range(min(limit, total)):
            raw_id = str(ds.ids[i])
            clean_id = (raw_id[:50] + "...") if len(raw_id) > 50 else raw_id
            console.print(f" - {clean_id}", markup=False)
            
    except Exception as e:
        console.print(f"[{BOLD_STYLE}{RED_STYLE}]Error loading Gold Layer:[/{BOLD_STYLE}{RED_STYLE}] {str(e)}", markup=False)


def main():
    app()


if __name__ == "__main__":
    main()