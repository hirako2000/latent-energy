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

PARQUET_FILENAME = "test.parquet"
SILVER_SUFFIX = "_clean.parquet"
VIZ_DIR_NAME = "visualizations"
SILVER_METRICS_JSON = "silver_metrics.json"

SMALL_SIZE_MAX = 5
MEDIUM_SIZE_MAX = 8
LARGE_SIZE = 12
SIZE_LABELS = ["5x5", "8x8", "12x12", "other"]

SIZE_COLORS = {
    "5x5": "#4e79a7",
    "8x8": "#f28e2c", 
    "12x12": "#59a14f",
    "other": "#e15759"
}

# For Plotting
FIGURE_SIZE_LARGE = (10, 6)
FIGURE_SIZE_MEDIUM = (10, 5)
DPI_HIGH = 150
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS = 12
FONT_WEIGHT_BOLD = "bold"
BAR_ANNOTATION_FONT_SIZE = 10
BINS_COUNT = 15
ALPHA_GRID = 0.3
SCATTER_ALPHA = 0.7
SCATTER_SIZE = 60
SCATTER_EDGE_WIDTH = 0.5
HISTPLOT_COLOR = "crimson"

PROGRESS_DESCRIPTION_TRAIN = "[yellow]Refining Medallion Tiers..."
PROGRESS_DESCRIPTION_SUBSET = "[magenta]  {subset_name}"

HINT_VALUE_THRESHOLD = 0

JSON_PATTERN = r"(\{.*\})"
ANSWER_KEY = "answer"
PERCEPTION_KEY = "perception"
GRID_KEY = "grid"

GRID_CELL_REPLACEMENT = "0"

SUCCESS_SYMBOL = "✓"
FAILURE_SYMBOL = "✗"
DIM_STYLE = "dim"
GREEN_STYLE = "green"
RED_STYLE = "red"
CYAN_STYLE = "cyan"
MAGENTA_STYLE = "magenta"
YELLOW_STYLE = "yellow"
BLUE_STYLE = "blue"


def normalize_grid(grid):
    if not grid or not isinstance(grid, list):
        return []
    return [[str(cell).replace('*', GRID_CELL_REPLACEMENT) for cell in row] for row in grid]


def robust_parse_solution(text):
    if not isinstance(text, str):
        return None
    
    match = re.search(JSON_PATTERN, text, re.DOTALL)
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
                    return {ANSWER_KEY: ast.literal_eval(ans_match.group(1))}
                except Exception:
                    pass
    return None


def extract_solution_grid(answer_data):
    if ANSWER_KEY in answer_data:
        return answer_data[ANSWER_KEY]
    elif PERCEPTION_KEY in answer_data:
        return answer_data[PERCEPTION_KEY]
    elif GRID_KEY in answer_data:
        return answer_data[GRID_KEY]
    return None


def classify_puzzle_size(size):
    if size <= SMALL_SIZE_MAX:
        return "5x5"
    elif size <= MEDIUM_SIZE_MAX:
        return "8x8"
    elif size == LARGE_SIZE:
        return "12x12"
    else:
        return "other"


def calculate_complexity(hint_dict):
    total = 0
    row_hints = hint_dict.get('row_hints', [])
    for hint_list in row_hints:
        if isinstance(hint_list, list):
            total += sum(hint_list)
    return total


def process_single_row(row, row_index):
    try:
        raw = row.to_dict()
        init_data = ast.literal_eval(raw['initialization'])
        answer_data = robust_parse_solution(raw['sample_answer'])
        
        if not answer_data:
            raise ValueError("Regex failed to find valid JSON/Dict in sample_answer")

        init_grid = init_data.get('initialization', [])
        sol_grid = extract_solution_grid(answer_data)
        
        if sol_grid is None:
            raise ValueError(f"No grid keys found. Keys: {list(answer_data.keys())}")

        raw_hints = init_data.get('hints', init_data)
        
        record_dict = {
            "id": str(raw.get('file_name', row_index)),
            "size": len(init_grid),
            "initialization": normalize_grid(init_grid),
            "solution": normalize_grid(sol_grid),
            "hints": {
                "row_hints": raw_hints.get('row_hints', []),
                "col_hints": raw_hints.get('col_hints', [])
            }
        }

        puzzle = msgspec.convert(record_dict, NonogramPuzzle)
        return msgspec.to_builtins(puzzle), None
        
    except Exception as e:
        return None, str(e)


def save_subset_results(subset_name, valid_records, first_error):
    if valid_records:
        silver_df = pd.DataFrame(valid_records)
        output_path = settings.SILVER_DIR / f"{subset_name}{SILVER_SUFFIX}"
        silver_df.to_parquet(output_path, engine='pyarrow', index=False)
        
        generate_atlas(silver_df, subset_name)
        console.print(f"[{GREEN_STYLE}]{SUCCESS_SYMBOL} {subset_name}: {len(valid_records)} rows refined.[/{GREEN_STYLE}]")
    else:
        console.print(f"[bold {RED_STYLE}]{FAILURE_SYMBOL} {subset_name} FAILED COMPLETELY[/bold {RED_STYLE}]")
        if first_error:
            console.print(f"[{DIM_STYLE}]First error: {first_error}[/{DIM_STYLE}]")


def process_silver_data():
    settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    subsets = [d for d in settings.BRONZE_DIR.iterdir() if d.is_dir() and (d / PARQUET_FILENAME).exists()]
    
    all_size_stats = {label: 0 for label in SIZE_LABELS}
    all_complexities = []
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        overall_task = progress.add_task(PROGRESS_DESCRIPTION_TRAIN, total=len(subsets))
        
        for subset_dir in subsets:
            subset_name = subset_dir.name
            df = pd.read_parquet(subset_dir / PARQUET_FILENAME)
            valid_records = []
            first_error = None

            row_task = progress.add_task(PROGRESS_DESCRIPTION_SUBSET.format(subset_name=subset_name), total=len(df))
            
            for i, (_, row) in enumerate(df.iterrows()):
                record, error = process_single_row(row, i)
                
                if record:
                    valid_records.append(record)
                    
                    size = record["size"]
                    size_key = classify_puzzle_size(size)
                    all_size_stats[size_key] += 1
                    
                    complexity = calculate_complexity(record["hints"])
                    all_complexities.append({
                        "size": size,
                        "complexity": complexity,
                        "subset": subset_name,
                        "size_label": f"{size}x{size}"
                    })
                elif first_error is None and error:
                    first_error = {"row": i, "error": error}
                    
                progress.advance(row_task)
            
            progress.remove_task(row_task)
            save_subset_results(subset_name, valid_records, first_error)
            progress.advance(overall_task)
    
    if all_complexities:
        generate_silver_visualizations(all_size_stats, all_complexities)


def generate_atlas(df, name):
    viz_dir = settings.ROOT_DIR / "docs" / VIZ_DIR_NAME
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    df['complexity'] = df['hints'].apply(calculate_complexity)
    
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    sns.histplot(data=df, x='complexity', kde=True, color=HISTPLOT_COLOR, bins=BINS_COUNT)
    plt.title(f'Complexity Atlas: {name}', fontsize=FONT_SIZE_TITLE, fontweight=FONT_WEIGHT_BOLD)
    plt.xlabel('Complexity (Sum of Row Hints)', fontsize=FONT_SIZE_AXIS)
    plt.ylabel('Frequency', fontsize=FONT_SIZE_AXIS)
    plt.grid(axis='y', alpha=ALPHA_GRID)
    plt.tight_layout()
    plt.savefig(viz_dir / f"{name}_complexity_atlas.png", dpi=DPI_HIGH)
    plt.close()


def generate_silver_visualizations(size_stats, complexities):
    viz_dir = settings.ROOT_DIR / "docs" / VIZ_DIR_NAME
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "size_distribution": size_stats,
        "complexities": complexities,
        "total_puzzles": sum(size_stats.values())
    }
    
    json_path = viz_dir / SILVER_METRICS_JSON
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"[{DIM_STYLE}]Metrics saved to: {json_path}[/{DIM_STYLE}]")
    
    generate_silver_pngs_from_metrics(metrics, viz_dir)
    print_silver_summary(metrics)


def generate_silver_pngs_from_metrics(metrics, viz_dir):
    sizes = SIZE_LABELS
    counts = [metrics["size_distribution"].get(s, 0) for s in sizes]
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    bars = plt.bar(sizes, counts, color=[SIZE_COLORS[s] for s in sizes])
    plt.title('Puzzle Size Distribution (Silver Layer)', fontsize=FONT_SIZE_TITLE, fontweight=FONT_WEIGHT_BOLD)
    plt.xlabel('Puzzle Size', fontsize=FONT_SIZE_AXIS)
    plt.ylabel('Number of Puzzles', fontsize=FONT_SIZE_AXIS)
    plt.grid(axis='y', alpha=ALPHA_GRID)
    
    for bar in bars:
        height = bar.get_height()
        if height > HINT_VALUE_THRESHOLD:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=BAR_ANNOTATION_FONT_SIZE)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "silver_size_distribution.png", dpi=DPI_HIGH)
    plt.close()
    
    if metrics["complexities"]:
        comp_df = pd.DataFrame(metrics["complexities"])
        
        plt.figure(figsize=FIGURE_SIZE_LARGE)
        
        for size_label in comp_df['size_label'].unique():
            subset = comp_df[comp_df['size_label'] == size_label]
            color = SIZE_COLORS.get(size_label, SIZE_COLORS["other"])
            
            plt.scatter(subset['size'], subset['complexity'],
                       color=color, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                       label=size_label, edgecolors='black', linewidth=SCATTER_EDGE_WIDTH)
        
        plt.title('Puzzle Complexity vs Size', fontsize=FONT_SIZE_TITLE, fontweight=FONT_WEIGHT_BOLD)
        plt.xlabel('Puzzle Size', fontsize=FONT_SIZE_AXIS)
        plt.ylabel('Complexity (Sum of Hints)', fontsize=FONT_SIZE_AXIS)
        plt.legend()
        plt.grid(True, alpha=ALPHA_GRID)
        plt.tight_layout()
        plt.savefig(viz_dir / "silver_complexity_vs_size.png", dpi=DPI_HIGH)
        plt.close()


def print_silver_summary(metrics):
    size_stats = metrics["size_distribution"]
    complexities = metrics["complexities"]
    
    table = Table(title="Silver Layer Dataset Composition", show_header=True, header_style=f"bold {CYAN_STYLE}")
    table.add_column("Puzzle Size", style=CYAN_STYLE, justify="center")
    table.add_column("Count", style=GREEN_STYLE, justify="right")
    table.add_column("% of Total", style=YELLOW_STYLE, justify="right")
    table.add_column("Avg Complexity", style=MAGENTA_STYLE, justify="right")
    
    total = metrics["total_puzzles"]
    
    for size_key in SIZE_LABELS:
        count = size_stats.get(size_key, 0)
        if count == 0:
            continue
        
        percentage = (count / total) * 100 if total > 0 else 0
        
        size_num = int(size_key.split('x')[0]) if size_key != "other" else 0
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
    console.print(Panel.fit(f"[bold {CYAN_STYLE}]Silver Layer Analysis Complete[/bold {CYAN_STYLE}]", border_style=CYAN_STYLE))
    console.print(table)
    console.print(f"\n[{DIM_STYLE}]Total puzzles processed: {total:,}")
    console.print(f"[{DIM_STYLE}]Visualizations saved to: docs/{VIZ_DIR_NAME}/[/{DIM_STYLE}]")