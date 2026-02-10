import pandas as pd
import numpy as np
import torch
from typing import Any, cast
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from grid_energy.utils.config import settings

from grid_energy.data.visualization.gold_visualizer import run_visualizer

console = Console()

MIN_SIZE_INCLUSIVE = 5
MAX_SIZE_INCLUSIVE = 5

GRID_DTYPE = np.int8
HINT_DTYPE = np.int16

DISPLAY_ID_MAX_LENGTH = 50
DISPLAY_ID_ELLIPSIS = "..."
PANEL_BORDER_STYLE = "cyan"
RANDOM_SAMPLE_COUNT = 1

def to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    return list(val)

def should_include_puzzle(size):
    return MIN_SIZE_INCLUSIVE <= size <= MAX_SIZE_INCLUSIVE

def analyze_dataset_dimensions(df):
    included_df = df[df['size'].apply(should_include_puzzle)]
    if len(included_df) == 0:
        return None, 0, 0
    
    max_grid = int(included_df['size'].max())
    max_hint_len = 0
    for h in included_df['hints']:
        all_h = to_list(h.get('row_hints', [])) + to_list(h.get('col_hints', []))
        for sub in all_h:
            if isinstance(sub, (list, np.ndarray)):
                max_hint_len = max(max_hint_len, len(sub))
    return included_df, max_grid, max_hint_len

def create_empty_tensors(num_samples, max_grid, max_hint_len):
    grid_tensors = np.zeros((num_samples, max_grid, max_grid), dtype=GRID_DTYPE)
    row_hints = np.zeros((num_samples, max_grid, max_hint_len), dtype=HINT_DTYPE)
    col_hints = np.zeros((num_samples, max_grid, max_hint_len), dtype=HINT_DTYPE)
    return grid_tensors, row_hints, col_hints

def process_puzzle_row(i, row, grid_tensors, row_hints, col_hints):
    sol = row['solution']
    curr_size = int(row['size'])
    
    # fill the grids
    for r_idx, grid_row in enumerate(sol[:curr_size]):
        for c_idx, cell in enumerate(grid_row[:curr_size]):
            if str(cell) == 's':
                grid_tensors[i, r_idx, c_idx] = 1
    
    h_dict = cast(Any, row['hints'])
    
    # row hints
    row_hint_lists = to_list(h_dict.get('row_hints', []))[:curr_size]
    for r_idx, h_list in enumerate(row_hint_lists):
        if isinstance(h_list, (list, np.ndarray)):
            clean_h = [v for v in h_list if v > 0]
            if clean_h:
                row_hints[i, r_idx, :len(clean_h)] = clean_h
    
    # col hints
    col_hint_lists = to_list(h_dict.get('col_hints', []))[:curr_size]
    for c_idx, h_list in enumerate(col_hint_lists):
        if isinstance(h_list, (list, np.ndarray)):
            clean_h = [v for v in h_list if v > 0]
            if clean_h:
                col_hints[i, c_idx, :len(clean_h)] = clean_h

def save_gold_dataset(grid_tensors, row_hints, col_hints, num_samples, max_grid, max_hint_len, size_stats):
    cleaned_ids = [f"puzzle_{i:04d}" for i in range(num_samples)]
    gold_data = {
        "grids": torch.from_numpy(grid_tensors),
        "row_hints": torch.from_numpy(row_hints),
        "col_hints": torch.from_numpy(col_hints),
        "ids": cleaned_ids,
        "metadata": {
            "max_grid_size": max_grid,
            "max_hint_len": max_hint_len,
            "samples": num_samples,
            "size_distribution": size_stats,
        }
    }
    save_path = settings.GOLD_DIR / "croissant_dataset.pt"
    torch.save(gold_data, save_path)
    return gold_data

def synthesize_gold():
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    silver_files = list(settings.SILVER_DIR.glob("*.parquet"))
    
    if not silver_files:
        console.print("[red]No silver files found.[/red]")
        return

    with Progress(SpinnerColumn(), TextColumn("[yellow]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        load_task = progress.add_task("Loading Silver subsets...", total=len(silver_files))
        df = pd.concat([pd.read_parquet(f) for f in silver_files], ignore_index=True)
        progress.advance(load_task, len(silver_files))
        
        included_df, max_grid, max_hint_len = analyze_dataset_dimensions(df)
        if included_df is None:
            console.print("[red]No 5x5 puzzles found in silver data![/red]")
            return
            
        num_samples = len(included_df)
        grid_tensors, row_hints, col_hints = create_empty_tensors(num_samples, max_grid, max_hint_len)

        processing_task = progress.add_task("Synthesizing Croissant Tensors...", total=num_samples)
        for i, (_, row) in enumerate(included_df.iterrows()):
            process_puzzle_row(i, row, grid_tensors, row_hints, col_hints)
            progress.advance(processing_task)

        size_stats = {"5x5": num_samples}
        gold_data = save_gold_dataset(grid_tensors, row_hints, col_hints, num_samples, max_grid, max_hint_len, size_stats)

    console.print("\n[bold green]✓ Gold Layer Synthesized![/bold green]")
    
    run_visualizer()
    
    verify_croissant(data=gold_data)

def verify_croissant(data=None):
    if not data:
        path = settings.GOLD_DIR / "croissant_dataset.pt"
        if not path.exists():
            return
        data = torch.load(path, weights_only=False)

    grids, r_hints, ids, metadata = data["grids"], data["row_hints"], data["ids"], data["metadata"]
    idx = torch.randint(0, len(grids), (RANDOM_SAMPLE_COUNT,)).item()
    
    raw_id = str(ids[idx])
    display_id = (raw_id[:DISPLAY_ID_MAX_LENGTH - len(DISPLAY_ID_ELLIPSIS)] + DISPLAY_ID_ELLIPSIS) if len(raw_id) > DISPLAY_ID_MAX_LENGTH else raw_id
    
    console.print(Panel(f"[bold cyan]Verification Sample ID:[/]\n{display_id}", expand=False, border_style=PANEL_BORDER_STYLE))
    
    size = metadata['max_grid_size']
    for i in range(size):
        hints = [int(h) for h in r_hints[idx, i].tolist() if h > 0]
        row_str = "".join(["[bold cyan]█ [/]" if grids[idx, i, j] == 1 else "[dim]· [/]" for j in range(size)])
        console.print(f"{' '.join(map(str, hints)).rjust(10)} │ {row_str}")

if __name__ == "__main__":
    synthesize_gold()