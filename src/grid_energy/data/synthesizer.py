import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from grid_energy.utils.config import settings

console = Console()

def to_list(val):
    if val is None: return []
    if isinstance(val, np.ndarray): return val.tolist()
    return list(val)

def synthesize_gold():
    """Converts Silver Parquets into fixed-shape Gold tensors - Croissant"""
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    silver_files = list(settings.SILVER_DIR.glob("*.parquet"))
    
    if not silver_files:
        console.print("[red]No silver files found. Perhaps run 'just inject-silver' first.[/red]")
        return

    size_stats = {"5x5": 0, "8x8": 0, "12x12": 0, "other": 0}
    fill_rates_by_size = {"5x5": [], "8x8": [], "12x12": [], "other": []}
    hint_lengths_by_size = {"5x5": [], "8x8": [], "12x12": [], "other": []}
    hint_values_by_size = {"5x5": [], "8x8": [], "12x12": [], "other": []}
    complexities_by_size = {"5x5": [], "8x8": [], "12x12": [], "other": []}

    grid_tensors = None

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
            
            if curr_size <= 5:
                size_key = "5x5"
            elif curr_size <= 8:
                size_key = "8x8"
            elif curr_size == 12:
                size_key = "12x12"
            else:
                size_key = "other"
            
            size_stats[size_key] += 1
            
            filled_cells = 0
            total_cells = 0
            for r_idx, grid_row in enumerate(sol[:curr_size]):
                for c_idx, cell in enumerate(grid_row[:curr_size]):
                    if str(cell) == 's': 
                        grid_tensors[i, r_idx, c_idx] = 1
                        filled_cells += 1
                    total_cells += 1
            
            if total_cells > 0:
                fill_rate = filled_cells / total_cells
                fill_rates_by_size[size_key].append(fill_rate)
            
            h_dict = row['hints']
            
            row_hint_lists = to_list(h_dict.get('row_hints', []))[:curr_size]
            for r_idx, h_list in enumerate(row_hint_lists):
                if isinstance(h_list, (list, np.ndarray)):
                    clean_h = [v for v in h_list if v > 0]
                    if clean_h:
                        row_hints[i, r_idx, :len(clean_h)] = clean_h
                        hint_lengths_by_size[size_key].append(len(clean_h))
                        hint_values_by_size[size_key].extend(clean_h)
            
            col_hint_lists = to_list(h_dict.get('col_hints', []))[:curr_size]
            for c_idx, h_list in enumerate(col_hint_lists):
                if isinstance(h_list, (list, np.ndarray)):
                    clean_h = [v for v in h_list if v > 0]
                    if clean_h:
                        col_hints[i, c_idx, :len(clean_h)] = clean_h
                        hint_lengths_by_size[size_key].append(len(clean_h))
                        hint_values_by_size[size_key].extend(clean_h)
            
            complexity = 0
            for hint_list in row_hint_lists + col_hint_lists:
                if isinstance(hint_list, (list, np.ndarray)):
                    complexity += sum(hint_list)
            complexities_by_size[size_key].append(complexity)
            
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
                "samples": num_samples,
                "size_distribution": size_stats,
            }
        }
        
        save_path = settings.GOLD_DIR / "croissant_dataset.pt"
        torch.save(gold_data, save_path)
        progress.advance(save_task)

    generate_gold_heatmap(grid_tensors)
    
    generate_gold_metrics_and_visualizations(
        grid_tensors, size_stats, fill_rates_by_size, 
        hint_lengths_by_size, hint_values_by_size, complexities_by_size
    )
    
    console.print(f"\n[bold green]✓ Gold Layer Synthesized![/bold green]")
    
    verify_croissant(data=gold_data)

def generate_gold_heatmap(grids: np.ndarray):
    """Heatmap of cell density"""
    docs_dir = settings.ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    density = np.mean(grids, axis=0) 
    global_fill_rate = np.mean(grids)
    
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

def generate_gold_metrics_and_visualizations(grids: np.ndarray, size_stats: dict, 
                                           fill_rates: dict, hint_lengths: dict, 
                                           hint_values: dict, complexities: dict):
    viz_dir = settings.ROOT_DIR / "docs" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "size_distribution": size_stats,
        "size_metrics": {},
        "tensor_info": {
            "shape": list(grids.shape),
            "total_puzzles": len(grids),
            "total_cells": int(grids.size),
            "filled_cells": int(np.count_nonzero(grids)),
            "global_fill_rate": float(np.mean(grids))
        }
    }
    
    for size_key in ["5x5", "8x8", "12x12"]:
        if size_key in size_stats and size_stats[size_key] > 0:
            metrics["size_metrics"][size_key] = {
                "count": size_stats[size_key],
                "avg_fill_rate": float(np.mean(fill_rates[size_key])) if fill_rates[size_key] else 0,
                "avg_complexity": float(np.mean(complexities[size_key])) if complexities[size_key] else 0,
                "avg_hint_length": float(np.mean(hint_lengths[size_key])) if hint_lengths[size_key] else 0,
                "max_hint_value": float(np.max(hint_values[size_key])) if hint_values[size_key] else 0
            }
    
    json_path = viz_dir / "gold_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"[dim]Metrics saved to: {json_path}[/dim]")
    
    generate_gold_pngs_from_metrics(metrics, viz_dir)
    print_gold_summary_table(metrics)

def generate_gold_pngs_from_metrics(metrics: dict, viz_dir: Path):
    """Generate PNG visualizations from metrics"""
    size_stats = metrics["size_distribution"]
    size_metrics = metrics["size_metrics"]
    
    # Distribution
    sizes = ["5x5", "8x8", "12x12"]
    counts = [size_stats.get(s, 0) for s in sizes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sizes, counts, color=['#4e79a7', '#f28e2c', '#59a14f'])
    plt.title('Puzzle Size Distribution (Gold Layer)', fontsize=14, fontweight='bold')
    plt.xlabel('Puzzle Size', fontsize=12)
    plt.ylabel('Number of Puzzles', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "gold_size_distribution.png", dpi=150)
    plt.close()
    
    # Rate by size
    if size_metrics:
        sizes_list = []
        fill_rates = []
        complexities = []
        
        for size_key in ["5x5", "8x8", "12x12"]:
            if size_key in size_metrics:
                sizes_list.append(size_key)
                fill_rates.append(size_metrics[size_key]["avg_fill_rate"] * 100)
                complexities.append(size_metrics[size_key]["avg_complexity"])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sizes_list, fill_rates, color=['#4e79a7', '#f28e2c', '#59a14f'])
        plt.title('Average Fill Rate by Puzzle Size', fontsize=14, fontweight='bold')
        plt.xlabel('Puzzle Size', fontsize=12)
        plt.ylabel('Fill Rate (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "gold_fill_rate_by_size.png", dpi=150)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sizes_list, complexities, color=['#e377c2', '#17becf', '#bcbd22'])
        plt.title('Average Complexity by Puzzle Size', fontsize=14, fontweight='bold')
        plt.xlabel('Puzzle Size', fontsize=12)
        plt.ylabel('Complexity (Sum of Hints)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "gold_complexity_by_size.png", dpi=150)
        plt.close()

def print_gold_summary_table(metrics: dict):
    size_stats = metrics["size_distribution"]
    size_metrics = metrics["size_metrics"]
    tensor = metrics["tensor_info"]
    
    table = Table(title="Gold Layer Synthesis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Size", style="cyan", justify="center")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Avg Fill", style="blue", justify="right")
    table.add_column("Avg Complexity", style="magenta", justify="right")
    table.add_column("Avg Hint Len", style="yellow", justify="right")
    table.add_column("Max Hint Val", style="red", justify="right")
    
    total_puzzles = sum(size_stats.values())
    
    for size_key in ["5x5", "8x8", "12x12"]:
        if size_key in size_metrics:
            m = size_metrics[size_key]
            table.add_row(
                size_key,
                f"{m['count']:,}",
                f"{m['avg_fill_rate']*100:.1f}%",
                f"{m['avg_complexity']:.1f}",
                f"{m['avg_hint_length']:.1f}",
                f"{m['max_hint_value']:.0f}"
            )
    
    console.print("\n")
    console.print(Panel.fit("[bold magenta]Gold Layer Synthesis Complete[/bold magenta]", border_style="magenta"))
    console.print(table)
    
    info_table = Table(show_header=False, box=None)
    info_table.add_row(f"[dim]Total puzzles:[/dim] [bold]{total_puzzles:,}[/bold]")
    info_table.add_row(f"[dim]Global fill rate:[/dim] [bold]{tensor['global_fill_rate']:.2%}[/bold]")
    info_table.add_row(f"[dim]Tensor shape:[/dim] [bold]{tensor['shape']}[/bold]")
    info_table.add_row(f"[dim]JSON metrics:[/dim] [bold]docs/visualizations/gold_metrics.json[/bold]")
    
    console.print("\n")
    console.print(info_table)

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
    c_hints = data["col_hints"]
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

    # puzzle size by checking hints
    max_size = metadata['max_grid_size']
    actual_size = max_size
    
    # last row/col with non-zero hints
    last_row_with_hints = -1
    last_col_with_hints = -1
    
    for i in range(max_size):
        if torch.any(r_hints[idx, i] > 0):
            last_row_with_hints = i
        if torch.any(c_hints[idx, i] > 0):
            last_col_with_hints = i
    
    if last_row_with_hints >= 0 and last_col_with_hints >= 0:
        actual_size = max(last_row_with_hints, last_col_with_hints) + 1
    elif last_row_with_hints >= 0:
        actual_size = last_row_with_hints + 1
    elif last_col_with_hints >= 0:
        actual_size = last_col_with_hints + 1
    
    console.print(f"[dim]Puzzle size: {actual_size}x{actual_size}[/dim]")
    
    console.print("[dim]Hints      │ Grid Layout[/]")
    console.print("[dim]───────────┼" + "──" * actual_size + "[/]")
    
    for i in range(min(actual_size, 8)):
        hints = [int(h) for h in r_hints[idx, i].tolist() if h > 0]
        hints_str = " ".join(map(str, hints)).rjust(10)
        
        row_str = ""
        for j in range(min(actual_size, 12)):
            row_str += "[bold cyan]█ [/]" if grid[i, j] == 1 else "[dim]· [/]"
        
        console.print(f"{hints_str} │ {row_str}")