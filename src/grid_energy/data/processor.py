import ast
import re
import json
import pandas as pd
import msgspec
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from grid_energy.utils.config import settings
from grid_energy.schemas.nonogram import NonogramPuzzle

console = Console()

def normalize_grid(grid):
    if not grid or not isinstance(grid, list):
        return []
    return [[str(cell).replace('*', '0') for cell in row] for row in grid]

def robust_parse_solution(text: str):
    if not isinstance(text, str):
        return None
    
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
        
    content = match.group(1).strip()
    
    try:
        return json.loads(content)
    except Exception:
        try:
            return ast.literal_eval(content)
        except Exception:
            ans_match = re.search(r'"answer":\s*(\[\[.*?\]\])', content, re.DOTALL)
            if ans_match:
                try:
                    return {"answer": ast.literal_eval(ans_match.group(1))}
                except Exception:
                    console.print("Ouch")
                    pass
    return None

def process_silver_data():
    settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    subsets = [d for d in settings.BRONZE_DIR.iterdir() if d.is_dir() and (d / "test.parquet").exists()]
    
    all_size_stats = {"5x5": 0, "8x8": 0, "12x12": 0, "other": 0}
    all_complexities = []
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        overall_task = progress.add_task("[yellow]Refining Medallion Tiers...", total=len(subsets))
        
        for subset_dir in subsets:
            subset_name = subset_dir.name
            df = pd.read_parquet(subset_dir / "test.parquet")
            valid_records = []
            first_error = None

            row_task = progress.add_task(f"[magenta]  {subset_name}", total=len(df))
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    raw = row.to_dict()
                    init_data = ast.literal_eval(raw['initialization'])
                    answer_data = robust_parse_solution(raw['sample_answer'])
                    
                    if not answer_data:
                        raise ValueError("Regex failed to find valid JSON/Dict in sample_answer")

                    init_grid = init_data.get('initialization', [])
                    sol_grid = answer_data.get('answer') or answer_data.get('perception') or answer_data.get('grid')
                    
                    if sol_grid is None:
                        raise ValueError(f"No grid keys found. Keys: {list(answer_data.keys())}")

                    raw_hints = init_data.get('hints', init_data)
                    
                    record_dict = {
                        "id": str(raw.get('file_name', i)),
                        "size": len(init_grid),
                        "initialization": normalize_grid(init_grid),
                        "solution": normalize_grid(sol_grid),
                        "hints": {
                            "row_hints": raw_hints.get('row_hints', []),
                            "col_hints": raw_hints.get('col_hints', [])
                        }
                    }

                    puzzle = msgspec.convert(record_dict, NonogramPuzzle)
                    valid_records.append(msgspec.to_builtins(puzzle))
                    
                    size = len(init_grid)
                    if size <= 5:
                        all_size_stats["5x5"] += 1
                    elif size <= 8:
                        all_size_stats["8x8"] += 1
                    elif size == 12:
                        all_size_stats["12x12"] += 1
                    else:
                        all_size_stats["other"] += 1
                    
                    complexity = 0
                    for hint_list in record_dict["hints"]["row_hints"] + record_dict["hints"]["col_hints"]:
                        if isinstance(hint_list, list):
                            complexity += sum(hint_list)
                    
                    all_complexities.append({
                        "size": size,
                        "complexity": complexity,
                        "subset": subset_name,
                        "size_label": f"{size}x{size}"
                    })

                except Exception as e:
                    if first_error is None:
                        first_error = {"row": i, "error": str(e)}
                    continue
                finally:
                    progress.advance(row_task)
            
            progress.remove_task(row_task)
            
            if valid_records:
                silver_df = pd.DataFrame(valid_records)
                output_path = settings.SILVER_DIR / f"{subset_name}_clean.parquet"
                silver_df.to_parquet(output_path, engine='pyarrow', index=False)
                
                generate_atlas(silver_df, subset_name)
                console.print(f"[green]✓ {subset_name}: {len(valid_records)} rows refined.")
            else:
                console.print(f"[bold red]✗ {subset_name} FAILED COMPLETELY[/bold red]")
                if first_error:
                    console.print(f"[dim]First error: {first_error['error']}[/dim]")
            
            progress.advance(overall_task)
    
    if all_complexities:
        generate_silver_visualizations(all_size_stats, all_complexities)

def generate_atlas(df: pd.DataFrame, name: str):
    viz_dir = settings.ROOT_DIR / "docs" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    def calc_complexity(h):
        total = 0
        row_hints = h.get('row_hints', [])
        for hint_list in row_hints:
            if isinstance(hint_list, list):
                total += sum(hint_list)
        return total
    
    df['complexity'] = df['hints'].apply(calc_complexity)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='complexity', kde=True, color='crimson', bins=15)
    plt.title(f'Complexity Atlas: {name}', fontsize=14, fontweight='bold')
    plt.xlabel('Complexity (Sum of Row Hints)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / f"{name}_complexity_atlas.png", dpi=150)
    plt.close()

def generate_silver_visualizations(size_stats: dict, complexities: list):
    """From collected metrics"""
    viz_dir = settings.ROOT_DIR / "docs" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "size_distribution": size_stats,
        "complexities": complexities,
        "total_puzzles": sum(size_stats.values())
    }
    
    json_path = viz_dir / "silver_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"[dim]Metrics saved to: {json_path}[/dim]")
    
    generate_silver_pngs_from_metrics(metrics, viz_dir)
    print_silver_summary(metrics)

def generate_silver_pngs_from_metrics(metrics: dict, viz_dir: Path):
    sizes = ["5x5", "8x8", "12x12", "other"]
    counts = [metrics["size_distribution"].get(s, 0) for s in sizes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sizes, counts, color=['#4e79a7', '#f28e2c', '#59a14f', '#e15759'])
    plt.title('Puzzle Size Distribution (Silver Layer)', fontsize=14, fontweight='bold')
    plt.xlabel('Puzzle Size', fontsize=12)
    plt.ylabel('Number of Puzzles', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "silver_size_distribution.png", dpi=150)
    plt.close()
    
    if metrics["complexities"]:
        comp_df = pd.DataFrame(metrics["complexities"])
        
        plt.figure(figsize=(10, 6))
        
        size_colors = {'5x5': '#4e79a7', '8x8': '#f28e2c', '12x12': '#59a14f'}
        
        for size_label in comp_df['size_label'].unique():
            subset = comp_df[comp_df['size_label'] == size_label]
            color = size_colors.get(size_label, '#e15759')
            
            plt.scatter(subset['size'], subset['complexity'],
                       color=color, alpha=0.7, s=60,
                       label=size_label, edgecolors='black', linewidth=0.5)
        
        plt.title('Puzzle Complexity vs Size', fontsize=14, fontweight='bold')
        plt.xlabel('Puzzle Size', fontsize=12)
        plt.ylabel('Complexity (Sum of Hints)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "silver_complexity_vs_size.png", dpi=150)
        plt.close()

def print_silver_summary(metrics: dict):
    size_stats = metrics["size_distribution"]
    complexities = metrics["complexities"]
    
    table = Table(title="Silver Layer Dataset Composition", show_header=True, header_style="bold cyan")
    table.add_column("Puzzle Size", style="cyan", justify="center")
    table.add_column("Count", style="green", justify="right")
    table.add_column("% of Total", style="yellow", justify="right")
    table.add_column("Avg Complexity", style="magenta", justify="right")
    
    total = metrics["total_puzzles"]
    
    for size_key in ["5x5", "8x8", "12x12"]:
        count = size_stats.get(size_key, 0)
        if count == 0:
            continue
        
        percentage = (count / total) * 100
        
        size_num = int(size_key.split('x')[0])
        size_complexities = [c["complexity"] for c in complexities if c["size"] == size_num]
        
        if size_complexities:
            avg_complexity = sum(size_complexities) / len(size_complexities)
        else:
            avg_complexity = 0
        
        table.add_row(
            size_key,
            f"{count:,}",
            f"{percentage:.1f}%",
            f"{avg_complexity:.1f}"
        )
    
    console.print("\n")
    console.print(Panel.fit("[bold cyan]Silver Layer Analysis Complete[/bold cyan]", border_style="cyan"))
    console.print(table)
    console.print(f"\n[dim]Total puzzles processed: {total:,}")
    console.print("[dim]Visualizations saved to: docs/visualizations/[/dim]")
