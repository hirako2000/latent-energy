import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Any, cast
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from grid_energy.utils.config import settings

console = Console()

VIZ_DIR_NAME = "visualizations"
GOLD_METRICS_JSON = "gold_metrics.json"
GOLD_HEATMAP_NAME = "gold_occupancy_heatmap.png"
GOLD_SUMMARY_PNG_NAME = "gold_summary_metrics.png"

def to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    return list(val)

def generate_gold_occupancy_heatmap(grids, viz_dir):
    density = np.mean(grids, axis=0)
    global_fill_rate = np.mean(grids)

    plt.figure(figsize=(8, 6))
    sns.heatmap(density, annot=True, fmt=".2f", cmap="magma")
    plt.title(f"Cell Occupancy Map (Global Fill Rate: {global_fill_rate:.2%})")
    plt.xlabel("Column Index (0-4)")
    plt.ylabel("Row Index (0-4)")
    
    
    
    plt.savefig(viz_dir / GOLD_HEATMAP_NAME)
    plt.close()

def generate_gold_summary_metrics_png(metrics, viz_dir):
    """Renders the summary data as a clean visual table image."""
    m = metrics["size_metrics"]["5x5"]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    
    data = [
        ["Metric", "Value"],
        ["Total Puzzles", f"{m['count']:,}"],
        ["Avg Fill Rate", f"{m['avg_fill_rate']*100:.1f}%"],
        ["Avg Complexity (Hint Sum)", f"{m['avg_complexity']:.1f}"],
        ["Avg Active Hint Lines", f"{m['avg_hint_count']:.1f}"],
        ["Max Hint Value", f"{m['max_hint_value']:.0f}"]
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='left', colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4e79a7')
            
    plt.title("Gold Layer Data Synthesis Summary", pad=20, fontsize=14, fontweight='bold')
    plt.savefig(viz_dir / GOLD_SUMMARY_PNG_NAME, dpi=150, bbox_inches='tight')
    plt.close()

def run_visualizer():
    silver_files = list(settings.SILVER_DIR.glob("*.parquet"))
    if not silver_files:
        console.print("[red]No silver files found.[/red]")
        return

    grid_acc, complexities, hint_counts, fill_rates, hint_vals = [], [], [], [], []

    with Progress(SpinnerColumn(), TextColumn("[yellow]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        load_task = progress.add_task("Loading Silver data...", total=len(silver_files))
        df = pd.concat([pd.read_parquet(f) for f in silver_files], ignore_index=True)
        progress.advance(load_task, len(silver_files))

        included_df = df[df['size'] == 5].copy()
        if len(included_df) == 0:
            console.print("[red]No 5x5 puzzles found.[/red]")
            return

        process_task = progress.add_task("Analyzing Gold Distribution...", total=len(included_df))
        
        for _, row in included_df.iterrows():
            sol = row['solution']
            temp_grid = np.zeros((5, 5), dtype=np.int8)
            filled = 0
            for r_idx, grid_row in enumerate(sol[:5]):
                for c_idx, cell in enumerate(grid_row[:5]):
                    if str(cell) == 's':
                        temp_grid[r_idx, c_idx] = 1
                        filled += 1
            grid_acc.append(temp_grid)
            fill_rates.append(filled / 25)

            h_dict = cast(Any, row['hints'])
            p_comp, p_active = 0, 0
            for h_list in (to_list(h_dict.get('row_hints', [])) + to_list(h_dict.get('col_hints', []))):
                clean = [v for v in h_list if v > 0] if isinstance(h_list, (list, np.ndarray)) else []
                if clean:
                    p_active += 1
                    hint_vals.extend(clean)
                    p_comp += sum(clean)
            
            complexities.append(p_comp)
            hint_counts.append(p_active)
            progress.advance(process_task)

    grids_np = np.array(grid_acc)
    viz_dir = settings.ROOT_DIR / "docs" / VIZ_DIR_NAME
    viz_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "size_metrics": {
            "5x5": {
                "count": len(included_df),
                "avg_fill_rate": float(np.mean(fill_rates)),
                "avg_complexity": float(np.mean(complexities)),
                "avg_hint_count": float(np.mean(hint_counts)),
                "max_hint_value": float(np.max(hint_vals))
            }
        },
        "tensor_info": {
            "shape": list(grids_np.shape),
            "global_fill_rate": float(np.mean(grids_np))
        }
    }

    generate_gold_occupancy_heatmap(grids_np, viz_dir)
    generate_gold_summary_metrics_png(metrics, viz_dir)
    
    with open(viz_dir / GOLD_METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)

    console.print(f"\n[bold green]✓ Visualizations generated in {viz_dir}[/bold green]")

if __name__ == "__main__":
    run_visualizer()