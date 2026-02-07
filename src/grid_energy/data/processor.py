import ast
import re
import json
import pandas as pd
import msgspec
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from grid_energy.utils.config import settings
from grid_energy.schemas.nonogram import NonogramPuzzle

console = Console()

def normalize_grid(grid):
    if not grid or not isinstance(grid, list): return []
    # cells are 's' or 'e' - typical in 5x5/8x8
    return [[str(cell).replace('*', '0') for cell in row] for row in grid]

def robust_parse_solution(text: str):
    if not isinstance(text, str): return None
    
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
        
    content = match.group(1).strip()
    
    try:
        return json.loads(content)
    except:
        try:
            return ast.literal_eval(content)
        except:
            ans_match = re.search(r'"answer":\s*(\[\[.*?\]\])', content, re.DOTALL)
            if ans_match:
                try:
                    return {"answer": ast.literal_eval(ans_match.group(1))}
                except: pass
    return None

def process_silver_data():
    settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    subsets = [d for d in settings.BRONZE_DIR.iterdir() if d.is_dir() and (d / "test.parquet").exists()]
    
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
                    # answer > perception > grid
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

                except Exception as e:
                    if first_error is None:
                        first_error = {"row": i, "error": str(e), "data": record_dict if 'record_dict' in locals() else "No data"}
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
                    t = Table(title=f"Debug Autopsy: {subset_name}")
                    t.add_column("Key", style="cyan")
                    t.add_column("Value", style="magenta")
                    t.add_row("Error", first_error['error'])
                    console.print(t)
            
            progress.advance(overall_task)

def generate_atlas(df: pd.DataFrame, name: str):
    docs_dir = settings.ROOT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    def calc_complexity(h):
        return sum(sum(r) if isinstance(r, list) else 0 for r in h.get('row_hints', []))
    df['complexity'] = df['hints'].apply(calc_complexity)
    sns.histplot(data=df, x='complexity', kde=True, color='crimson')
    plt.title(f"Complexity Atlas: {name}")
    plt.savefig(docs_dir / f"{name}_complexity.png")
    plt.close()