import torch
import torch.optim as optim
import torch.nn.functional as F
from rich.live import Live
from rich.table import Table
from rich.progress import Progress
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import Dataset, DataLoader

console = Console()

def train_ebm(epochs: int, batch_size: int, lr: float, validation_freq: int = 5, val_size: float = 0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    from grid_energy.core.models import NonogramCNN
    from grid_energy.core.energy import NonogramEnergy
    from grid_energy.core.solver import KineticSolver
    from grid_energy.data.loader import CroissantDataset
    from torch.utils.data import Subset

    full_dataset = CroissantDataset()
    
    # train/val splits
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=42, shuffle=True
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    console.print(f"[dim]Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation[/dim]")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    backbone = NonogramCNN(grid_size=12).to(device)
    energy_fn = NonogramEnergy(backbone).to(device)
    solver = KineticSolver(energy_fn, n_steps=150, deterministic=True)
    
    optimizer = optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-5)
    
    # a rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    stats_table = Table(title=f"EBM Training ({device})")
    stats_table.add_column("Epoch", justify="center")
    stats_table.add_column("Loss", justify="right", style="cyan")
    stats_table.add_column("Sol E", justify="right", style="green")
    stats_table.add_column("Noise E", justify="right", style="magenta")
    stats_table.add_column("LogicErr", justify="right", style="red")
    stats_table.add_column("Val Acc", justify="right", style="yellow")
    stats_table.add_column("5x5", justify="right", style="blue")
    stats_table.add_column("8x8", justify="right", style="blue")
    stats_table.add_column("12x12", justify="right", style="blue")
    stats_table.add_column("LR", justify="right", style="dim")

    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
    with Live(stats_table, refresh_per_second=2):
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_sol_energy = 0.0
            epoch_noise_energy = 0.0
            epoch_logic = 0.0
            total_batches = 0
            
            backbone.train()
            
            for batch in train_loader:
                pos_grid = batch["target_grid"].to(device).float()
                hints = batch["hints"].to(device)
                
                pos_continuous = (pos_grid - 0.5) * 2.0
                
                energy_fn.set_context(hints)
                
                optimizer.zero_grad()
                
                e_solution = energy_fn(pos_continuous)
                
                noise_grid = torch.randn_like(pos_continuous) * 0.3
                e_noise = energy_fn(noise_grid)
                
                margin = 1.0
                loss = F.relu(e_solution - e_noise + margin).mean()
                
                # push solution energy to be negative
                reg_loss = F.relu(e_solution + 0.5).mean()
                
                total_loss = loss + reg_loss * 0.1
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 0.5)
                optimizer.step()
                
                logic_err = energy_fn.check_logic(pos_continuous, hints)
                
                epoch_loss += total_loss.item()
                epoch_sol_energy += e_solution.mean().item()
                epoch_noise_energy += e_noise.mean().item()
                epoch_logic += logic_err
                total_batches += 1
            
            avg_loss = epoch_loss / total_batches
            avg_sol_energy = epoch_sol_energy / total_batches
            avg_noise_energy = epoch_noise_energy / total_batches
            avg_logic = epoch_logic / total_batches
            
            # adjust learning rate based on loss
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # some validation, periodically
            val_acc = None
            size_accuracies = {"5x5": None, "8x8": None, "12x12": None}
            
            if (epoch + 1) % validation_freq == 0 or epoch == 0:
                val_acc, size_accuracies = validate_model(
                    backbone, energy_fn, solver, val_loader, device
                )
                
                # keep best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(backbone.state_dict(), "data/models/ebm_best.pt")
                    console.print(f"[green]Saved new best model with val_acc: {val_acc:.2%}[/green]")
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            row_values = [
                f"{epoch+1:02d}",
                f"{avg_loss:.3f}",
                f"{avg_sol_energy:.3f}",
                f"{avg_noise_energy:.3f}",
                f"{avg_logic:.3f}",
            ]
            
            if val_acc is not None:
                val_acc_str = f"{val_acc:.2%}"
                if val_acc > 0.9:
                    val_acc_str = f"[green]{val_acc_str}[/green]"
                elif val_acc > 0.7:
                    val_acc_str = f"[yellow]{val_acc_str}[/yellow]"
                else:
                    val_acc_str = f"[red]{val_acc_str}[/red]"
                
                row_values.append(val_acc_str)
                
                for size_key in ["5x5", "8x8", "12x12"]:
                    acc = size_accuracies[size_key]
                    if acc is not None:
                        acc_str = f"{acc:.2%}"
                        if acc > 0.9:
                            acc_str = f"[green]{acc_str}[/green]"
                        elif acc > 0.7:
                            acc_str = f"[yellow]{acc_str}[/yellow]"
                        else:
                            acc_str = f"[red]{acc_str}[/red]"
                        row_values.append(acc_str)
                    else:
                        row_values.append("-")
            else:
                row_values.extend(["-", "-", "-", "-"])
            
            row_values.append(f"{current_lr:.1e}")
            stats_table.add_row(*row_values)
            
            # early stop
            if patience_counter >= patience and epoch > 30:
                console.print(f"\n[yellow]Early stopping at epoch {epoch+1}. No improvement for {patience} validations.[/yellow]")
                break
            
            # some pretty strict convergence check
            convergence_threshold = avg_loss < 0.05 and avg_sol_energy < (avg_noise_energy - 0.5)
            if epoch > 20 and convergence_threshold:
                console.print(f"\n[green]Convergence achieved at epoch {epoch+1}[/green]")
                if val_acc is None:
                    val_acc, size_accuracies = validate_model(
                        backbone, energy_fn, solver, val_loader, device
                    )
                console.print(f"Final validation accuracy: {val_acc:.2%}")
                break
    
    # now let's load the best model for some final test
    if best_val_acc > 0:
        backbone.load_state_dict(torch.load("data/models/ebm_best.pt", map_location=device))
        console.print(f"[green]Loaded best model (val_acc: {best_val_acc:.2%})[/green]")
    
    torch.save(backbone.state_dict(), "data/models/ebm_final.pt")
    
    console.print("\n[bold cyan]Running final test on entire dataset...[/bold cyan]")
    test_all_puzzles(backbone, energy_fn, solver, device, full_dataset)


def validate_model(backbone, energy_fn, solver, val_loader, device):
    backbone.eval()
    total_correct = 0
    total_samples = 0
    
    size_stats = {
        "5x5": {"correct": 0, "total": 0},
        "8x8": {"correct": 0, "total": 0},
        "12x12": {"correct": 0, "total": 0}
    }
    
    with torch.no_grad():
        for batch in val_loader:
            target_grid = batch["target_grid"].to(device).float()
            hints = batch["hints"].to(device)
            batch_size = target_grid.size(0)
            
            for b in range(batch_size):
                # one sample at a time for  memory efficiency
                single_target = target_grid[b:b+1]
                single_hints = hints[b:b+1]
                
                # inference!  todo add benchmarking
                init_state = torch.randn_like(single_target) * 0.1
                energy_fn.set_context(single_hints)
                resolved = solver.resolve(init_state, single_hints)
                
                # binary predictions
                pred_grid = (torch.sigmoid(resolved * 6.0) > 0.5).float()
                
                # puzzle size
                row_hints = single_hints[0, 0]
                col_hints = single_hints[0, 1]
                
                rows_with_hints = torch.where(row_hints.sum(dim=1) > 0)[0]
                cols_with_hints = torch.where(col_hints.sum(dim=1) > 0)[0]
                
                if len(rows_with_hints) > 0 and len(cols_with_hints) > 0:
                    size = max(rows_with_hints[-1].item(), cols_with_hints[-1].item()) + 1
                else:
                    size = 12
                
                #  some size category
                if size <= 5:
                    size_key = "5x5"
                elif size <= 8:
                    size_key = "8x8"
                else:
                    size_key = "12x12"
                
                # relevant area
                pred_puzzle = pred_grid[0, 0, :size, :size]
                target_puzzle = single_target[0, 0, :size, :size]
                
                # for puzzle to be correct, all pixels shall match
                is_correct = torch.all(pred_puzzle == target_puzzle).item()
                
                size_stats[size_key]["total"] += 1
                total_samples += 1
                
                if is_correct:
                    size_stats[size_key]["correct"] += 1
                    total_correct += 1
    
    backbone.train()
    
    # accuracies
    val_acc = total_correct / total_samples if total_samples > 0 else 0.0
    size_accuracies = {}
    
    for size_key in size_stats:
        if size_stats[size_key]["total"] > 0:
            size_accuracies[size_key] = size_stats[size_key]["correct"] / size_stats[size_key]["total"]
        else:
            size_accuracies[size_key] = None
    
    return val_acc, size_accuracies


def test_all_puzzles(backbone, energy_fn, solver, device, dataset):
    """Test the model on the entire dataset"""
    from grid_energy.data.loader import DataLoader
    
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    size_results = {
        "5x5": {"correct": 0, "total": 0, "failed_ids": []},
        "8x8": {"correct": 0, "total": 0, "failed_ids": []},
        "12x12": {"correct": 0, "total": 0, "failed_ids": []},
        "other": {"correct": 0, "total": 0, "failed_ids": []}
    }
    
    failed_examples = []
    
    backbone.eval()
    with torch.no_grad(), Progress() as progress:
        task = progress.add_task("[cyan]Testing all puzzles...", total=len(dataset))
        
        for batch in test_loader:
            hints = batch["hints"].to(device)
            target_grid = batch["target_grid"].to(device).float()
            puzzle_id = batch["id"][0]
            
            # puzzle size
            row_hints = hints[0, 0]
            col_hints = hints[0, 1]
            
            rows_with_hints = torch.where(row_hints.sum(dim=1) > 0)[0]
            cols_with_hints = torch.where(col_hints.sum(dim=1) > 0)[0]
            
            if len(rows_with_hints) > 0 and len(cols_with_hints) > 0:
                size = max(rows_with_hints[-1].item(), cols_with_hints[-1].item()) + 1
            else:
                size = 12
            
            # again the category
            if size <= 5:
                size_key = "5x5"
            elif size <= 8:
                size_key = "8x8"
            elif size == 12:
                size_key = "12x12"
            else:
                size_key = "other"
            
            size_results[size_key]["total"] += 1
            
            # again inference
            init_state = torch.randn_like(target_grid) * 0.1
            energy_fn.set_context(hints)
            resolved = solver.resolve(init_state, hints)
            pred_grid = (torch.sigmoid(resolved * 6.0) > 0.5).float()
            
            # correctness
            pred_puzzle = pred_grid[0, 0, :size, :size]
            target_puzzle = target_grid[0, 0, :size, :size]
            
            is_correct = torch.all(pred_puzzle == target_puzzle).item()
            
            if is_correct:
                size_results[size_key]["correct"] += 1
            else:
                size_results[size_key]["failed_ids"].append(puzzle_id)
                failed_examples.append({
                    "id": puzzle_id,
                    "size": size,
                    "pred": pred_puzzle.cpu().numpy(),
                    "target": target_puzzle.cpu().numpy(),
                    "hints": hints.cpu().numpy()
                })
            
            progress.update(task, advance=1)
    
    console.print("\n" + "="*60)
    console.print("[bold cyan]FINAL TEST RESULTS (All Puzzles)[/bold cyan]")
    console.print("="*60)
    
    total_correct = 0
    total_puzzles = 0
    
    for size_key in ["5x5", "8x8", "12x12", "other"]:
        stats = size_results[size_key]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            color = "green" if acc > 0.9 else "yellow" if acc > 0.7 else "red"
            
            console.print(f"\n[size_key]{size_key}:[/] [bold]{stats['correct']}/{stats['total']}[/] ({acc:.2%})")
            
            if stats["correct"] < stats["total"]:
                failed_count = stats["total"] - stats["correct"]
                console.print(f"  [dim]Failed: {failed_count} puzzles[/dim]")
                if stats["failed_ids"]:
                    console.print(f"  [dim]Example failed IDs: {', '.join(stats['failed_ids'][:3])}{'...' if len(stats['failed_ids']) > 3 else ''}[/dim]")
            
            total_correct += stats["correct"]
            total_puzzles += stats["total"]
    
    overall_acc = total_correct / total_puzzles
    console.print("\n" + "-"*60)
    console.print(f"[bold]OVERALL: {total_correct}/{total_puzzles} ({overall_acc:.2%})[/bold]")
    
    if overall_acc < 0.8:
        console.print("[yellow]⚠️  Model accuracy is below 80%. Consider:[/yellow]")
        console.print("  - Increasing training epochs")
        console.print("  - Adjusting learning rate")
        console.print("  - Adding more data augmentation")
        console.print("  - Tuning the energy function")
    elif overall_acc < 0.95:
        console.print("[yellow]⚠️  Model accuracy is good but could be improved[/yellow]")
    else:
        console.print("[green]✅ Excellent model accuracy![/green]")
    
    if failed_examples:
        failed_path = "data/models/failed_examples.pt"
        torch.save(failed_examples, failed_path)
        console.print(f"\n[yellow]Saved {len(failed_examples)} failed examples to {failed_path}[/yellow]")
        console.print("[dim]Use these to debug model failures[/dim]")
    
    return overall_acc