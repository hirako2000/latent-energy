import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from grid_energy.utils.config import settings

console = Console()

def to_list(val):
    """Safely convert numpy arrays or None types to lists for dimension analysis."""
    if val is None: return []
    if isinstance(val, np.ndarray): return val.tolist()
    return list(val)

def generate_gold_heatmap(grids: np.ndarray):
    """Generates a heatmap of cell density and prints distribution metrics to stdout."""
    docs_dir = settings.ROOT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # across the sample dimension (N, H, W) -> (H, W)
    density = np.mean(grids, axis=0) 
    global_fill_rate = np.mean(grids)
    
    # Per-cell entropy is : H = -sum(p * log2(p))
    p = np.clip(density, 1e-9, 1-1e-9) 
    entropy_map = -(p * np.log2(p) + (1-p) * np.log2(1-p))
    avg_entropy = np.mean(entropy_map)

    stats_table = Table(title="Gold Layer Data Health", show_header=True, header_style="bold cyan")
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    
    stats_table.add_row("Global Fill Rate (Sparsity)", f"{global_fill_rate:.2%}")
    stats_table.add_row("Dataset Entropy (Predictability)", f"{avg_entropy:.4f} bits")
    stats_table.add_row("Hot Spot (Max Density)", f"{np.max(density):.2%}")
    stats_table.add_row("Cold Spot (Min Density)", f"{np.min(density):.2%}")
    
    console.print(stats_table)

    plt.figure(figsize=(8, 6))
    sns.heatmap(density, annot=True, fmt=".2f", cmap="magma")
    plt.title(f"Cell Occupancy (Fill Rate: {global_fill_rate:.2%})")
    plt.savefig(docs_dir / "gold_occupancy_heatmap.png")
    plt.close()

def verify_croissant(data: dict = None):
    """
    ASCII
    """
    if data is None:
        path = settings.GOLD_DIR / "croissant_dataset.pt"
        if not path.exists():
            console.print("[red]Gold tensor not found! Run synthesis first.[/red]")
            return
        data = torch.load(path, weights_only=False)

    grids = data["grids"]
    r_hints = data["row_hints"]
    ids = data["ids"]
    metadata = data["metadata"]
    
    idx = torch.randint(0, len(grids), (1,)).item()
    grid = grids[idx]
    
    raw_id = str(ids[idx])
    display_id = (raw_id[:47] + "...") if len(raw_id) > 50 else raw_id
    
    
    console.print(Panel(
        f"[bold cyan]Verification Sample ID:[/]\n{display_id}",
        expand=False, 
        border_style="cyan",
        highlight=False
    ))

    size = metadata['max_grid_size']
    
    console.print("[dim]Hints      │ Grid Layout[/]")
    console.print("[dim]───────────┼" + "──" * size + "[/]")
    
    for i in range(size):
        hints = [int(h) for h in r_hints[idx, i].tolist() if h > 0]
        hints_str = " ".join(map(str, hints)).rjust(10)
        
        row_str = ""
        for j in range(size):
            row_str += "[bold cyan]█ [/]" if grid[i, j] == 1 else "[dim]· [/]"
        
        if any(grid[i]) or hints:
            console.print(f"{hints_str} │ {row_str}")

def synthesize_gold():
    """Converts Silver Parquets into fixed-shape Gold tensors - Croissant"""
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    silver_files = list(settings.SILVER_DIR.glob("*.parquet"))
    
    if not silver_files:
        console.print("[red]No silver files found. Perhaps run 'just inject-silver' first.[/red]")
        return

    grid_tensors = None
    gold_data = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[yellow]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        
        load_task = progress.add_task("Loading Silver subsets...", total=len(silver_files))
        df = pd.concat([pd.read_parquet(f) for f in silver_files], ignore_index=True)
        progress.advance(load_task, len(silver_files))
        
        analyze_task = progress.add_task("Analyzing dimensions...", total=1)
        max_grid = int(df['size'].max())
        max_hint_len = 0
        for h in df['hints']:
            all_h = to_list(h.get('row_hints', [])) + to_list(h.get('col_hints', []))
            for sub in all_h:
                if isinstance(sub, (list, np.ndarray)):
                    max_hint_len = max(max_hint_len, len(sub))
        progress.advance(analyze_task)

        num_samples = len(df)
        processing_task = progress.add_task("Synthesizing Croissant Tensors...", total=num_samples)
        
        grid_tensors = np.zeros((num_samples, max_grid, max_grid), dtype=np.int8)
        row_hints = np.zeros((num_samples, max_grid, max_hint_len), dtype=np.int16)
        col_hints = np.zeros((num_samples, max_grid, max_hint_len), dtype=np.int16)

        for i, row in df.iterrows():
            sol = row['solution']
            curr_size = int(row['size'])
            
            for r_idx, grid_row in enumerate(sol[:curr_size]):
                for c_idx, cell in enumerate(grid_row[:curr_size]):
                    if str(cell) == 's': 
                        grid_tensors[i, r_idx, c_idx] = 1

            h_dict = row['hints']
            for r_idx, h_list in enumerate(to_list(h_dict.get('row_hints', []))[:curr_size]):
                clean_h = [v for v in (h_list if isinstance(h_list, (list, np.ndarray)) else []) if v > 0]
                row_hints[i, r_idx, :len(clean_h)] = clean_h
            
            for c_idx, h_list in enumerate(to_list(h_dict.get('col_hints', []))[:curr_size]):
                clean_h = [v for v in (h_list if isinstance(h_list, (list, np.ndarray)) else []) if v > 0]
                col_hints[i, c_idx, :len(clean_h)] = clean_h
            
            progress.advance(processing_task)

        save_task = progress.add_task("Finalizing Gold Layer...", total=1)

       
        num_samples = len(df)
        cleaned_ids = [f"puzzle_{i:04d}" for i in range(num_samples)]

        gold_data = {
            "grids": torch.from_numpy(grid_tensors),
            "row_hints": torch.from_numpy(row_hints),
            "col_hints": torch.from_numpy(col_hints),
            "ids": cleaned_ids, 
            "metadata": {
                "max_grid_size": max_grid, 
                "max_hint_len": max_hint_len, 
                "samples": num_samples
            }
        }
        
        save_path = settings.GOLD_DIR / "croissant_dataset.pt"
        torch.save(gold_data, save_path)
        progress.advance(save_task)

    generate_gold_heatmap(grid_tensors)
    console.print(f"\n[bold green]✓ Gold Layer Synthesized![/bold green]")
    
    verify_croissant(data=gold_data)