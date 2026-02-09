import torch
import torch.nn.functional as F
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.table import Table
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Subset

from grid_energy.core.energy import NonogramEnergy
from grid_energy.core.models import NonogramCNN
from grid_energy.core.solver import KineticSolver
from grid_energy.data.loader import CroissantDataset, DataLoader

console = Console()

# Training constants
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VALIDATION_FREQUENCY = 5
DEFAULT_VAL_SIZE = 0.2
RANDOM_SEED = 42

# Model saving paths
BEST_MODEL_PATH = "data/models/ebm_best.pt"
FINAL_MODEL_PATH = "data/models/ebm_final.pt"
FAILED_EXAMPLES_PATH = "data/models/failed_examples.pt"

# Optimization constants
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 0.5
LEARNING_RATE_PLATEAU_FACTOR = 0.5
LEARNING_RATE_PLATEAU_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 15
CONVERGENCE_EPOCH_THRESHOLD = 20

# Energy and loss constants
ENERGY_MARGIN = 1.0
ENERGY_REGULARIZATION_THRESHOLD = 0.5
ENERGY_REGULARIZATION_WEIGHT = 0.1
CONVERGENCE_LOSS_THRESHOLD = 0.05
CONVERGENCE_ENERGY_DIFFERENCE = 0.5

# Inference constants
SIGMOID_SCALE = 6.0
SOLVER_STEPS = 150
INITIAL_NOISE_SCALE = 0.3
SOLVER_DETERMINISTIC = True

# Accuracy thresholds for reporting
LOW_ACCURACY_THRESHOLD = 0.8
HIGH_ACCURACY_THRESHOLD = 0.95

# Size categories (only 5x5 for gold)
SIZE_CATEGORIES = ["5x5"]


def create_optimization_components(model, learning_rate):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LEARNING_RATE_PLATEAU_FACTOR,
        patience=LEARNING_RATE_PLATEAU_PATIENCE
    )
    return optimizer, scheduler


def create_stats_table(device):
    stats_table = Table(title=f"EBM Training ({device})")
    stats_table.add_column("Epoch", justify="center")
    stats_table.add_column("Loss", justify="right", style="cyan")
    stats_table.add_column("Sol E", justify="right", style="green")
    stats_table.add_column("Noise E", justify="right", style="magenta")
    stats_table.add_column("LogicErr", justify="right", style="red")
    stats_table.add_column("Val Acc", justify="right", style="yellow")
    
    for size_key in SIZE_CATEGORIES:
        stats_table.add_column(size_key, justify="right", style="blue")
    
    stats_table.add_column("LR", justify="right", style="dim")
    return stats_table


def compute_training_loss(energy_fn, pos_continuous, noise_grid):
    e_solution = energy_fn(pos_continuous)
    e_noise = energy_fn(noise_grid)
    
    loss = F.relu(e_solution - e_noise + ENERGY_MARGIN).mean()
    reg_loss = F.relu(e_solution + ENERGY_REGULARIZATION_THRESHOLD).mean()
    
    total_loss = loss + reg_loss * ENERGY_REGULARIZATION_WEIGHT
    return total_loss, e_solution.mean().item(), e_noise.mean().item()


def train_epoch(model, energy_fn, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_sol_energy = 0.0
    epoch_noise_energy = 0.0
    epoch_logic = 0.0
    total_batches = 0

    for batch in train_loader:
        pos_grid = batch["target_grid"].to(device).float()
        hints = batch["hints"].to(device)
        
        pos_continuous = (pos_grid - 0.5) * 2.0
        energy_fn.set_context(hints)
        
        optimizer.zero_grad()
        
        noise_grid = torch.randn_like(pos_continuous) * INITIAL_NOISE_SCALE
        total_loss, sol_energy, noise_energy = compute_training_loss(
            energy_fn, pos_continuous, noise_grid
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        
        logic_err = energy_fn.check_logic(pos_continuous, hints)
        
        epoch_loss += total_loss.item()
        epoch_sol_energy += sol_energy
        epoch_noise_energy += noise_energy
        epoch_logic += logic_err
        total_batches += 1

    avg_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
    avg_sol_energy = epoch_sol_energy / total_batches if total_batches > 0 else 0.0
    avg_noise_energy = epoch_noise_energy / total_batches if total_batches > 0 else 0.0
    avg_logic = epoch_logic / total_batches if total_batches > 0 else 0.0
    
    return avg_loss, avg_sol_energy, avg_noise_energy, avg_logic


def format_accuracy_value(accuracy):
    if accuracy > HIGH_ACCURACY_THRESHOLD:
        return f"[green]{accuracy:.2%}[/green]"
    elif accuracy > LOW_ACCURACY_THRESHOLD:
        return f"[yellow]{accuracy:.2%}[/yellow]"
    else:
        return f"[red]{accuracy:.2%}[/red]"


def add_table_row(stats_table, epoch, avg_loss, avg_sol_energy, avg_noise_energy,
                 avg_logic, val_acc, size_accuracies, current_lr):
    row_values = [
        f"{epoch+1:02d}",
        f"{avg_loss:.3f}",
        f"{avg_sol_energy:.3f}",
        f"{avg_noise_energy:.3f}",
        f"{avg_logic:.3f}",
    ]

    if val_acc is not None:
        row_values.append(format_accuracy_value(val_acc))
        
        for size_key in SIZE_CATEGORIES:
            acc = size_accuracies.get(size_key)
            if acc is not None:
                row_values.append(format_accuracy_value(acc))
            else:
                row_values.append("-")
    else:
        row_values.extend(["-"] * (1 + len(SIZE_CATEGORIES)))

    row_values.append(f"{current_lr:.1e}")
    stats_table.add_row(*row_values)


def check_convergence(epoch, avg_loss, avg_sol_energy, avg_noise_energy):
    if epoch <= CONVERGENCE_EPOCH_THRESHOLD:
        return False
    
    loss_threshold_met = avg_loss < CONVERGENCE_LOSS_THRESHOLD
    energy_difference_met = avg_sol_energy < (avg_noise_energy - CONVERGENCE_ENERGY_DIFFERENCE)
    
    return loss_threshold_met and energy_difference_met


def validate_model(model, energy_fn, solver, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    size_stats = {size_key: {"correct": 0, "total": 0} for size_key in SIZE_CATEGORIES}
    
    with torch.no_grad():
        for batch in val_loader:
            target_grid = batch["target_grid"].to(device).float()
            hints = batch["hints"].to(device)
            batch_size = target_grid.size(0)

            for b in range(batch_size):
                single_target = target_grid[b:b+1]
                single_hints = hints[b:b+1]
                
                init_state = torch.randn_like(single_target) * INITIAL_NOISE_SCALE
                energy_fn.set_context(single_hints)
                resolved = solver.resolve(init_state, single_hints)
                
                pred_grid = (torch.sigmoid(resolved * SIGMOID_SCALE) > 0.5).float()
                
                row_hints = single_hints[0, 0]
                col_hints = single_hints[0, 1]
                
                rows_with_hints = torch.where(row_hints.sum(dim=1) > 0)[0]
                cols_with_hints = torch.where(col_hints.sum(dim=1) > 0)[0]
                
                if len(rows_with_hints) > 0 and len(cols_with_hints) > 0:
                    size = max(rows_with_hints[-1].item(), cols_with_hints[-1].item()) + 1
                else:
                    size = 12
                
                size_key = "5x5"
                
                pred_puzzle = pred_grid[0, 0, :size, :size]
                target_puzzle = single_target[0, 0, :size, :size]
                
                is_correct = torch.all(pred_puzzle == target_puzzle).item()
                
                size_stats[size_key]["total"] += 1
                total_samples += 1
                
                if is_correct:
                    size_stats[size_key]["correct"] += 1
                    total_correct += 1

    model.train()
    
    val_acc = total_correct / total_samples if total_samples > 0 else 0.0
    size_accuracies = {}
    
    for size_key in SIZE_CATEGORIES:
        if size_stats[size_key]["total"] > 0:
            size_accuracies[size_key] = size_stats[size_key]["correct"] / size_stats[size_key]["total"]
        else:
            size_accuracies[size_key] = None
    
    return val_acc, size_accuracies


def train_ebm(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
              lr=DEFAULT_LEARNING_RATE, validation_freq=DEFAULT_VALIDATION_FREQUENCY,
              val_size=DEFAULT_VAL_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    full_dataset = CroissantDataset()
    
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=RANDOM_SEED, shuffle=True
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
    
    backbone = NonogramCNN(grid_size=5).to(device)
    energy_fn = NonogramEnergy(backbone).to(device)
    solver = KineticSolver(energy_fn, n_steps=SOLVER_STEPS, deterministic=SOLVER_DETERMINISTIC)
    
    optimizer, scheduler = create_optimization_components(backbone, lr)
    stats_table = create_stats_table(device)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    with Live(stats_table, refresh_per_second=2):
        for epoch in range(epochs):
            avg_loss, avg_sol_energy, avg_noise_energy, avg_logic = train_epoch(
                backbone, energy_fn, train_loader, optimizer, device
            )
            
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            val_acc = None
            size_accuracies = {size_key: None for size_key in SIZE_CATEGORIES}
            
            if (epoch + 1) % validation_freq == 0 or epoch == 0:
                val_acc, size_accuracies = validate_model(
                    backbone, energy_fn, solver, val_loader, device
                )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(backbone.state_dict(), BEST_MODEL_PATH)
                    console.print(f"[green]Saved new best model with val_acc: {val_acc:.2%}[/green]")
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            add_table_row(
                stats_table, epoch, avg_loss, avg_sol_energy,
                avg_noise_energy, avg_logic, val_acc, size_accuracies, current_lr
            )
            
            if patience_counter >= EARLY_STOPPING_PATIENCE and epoch > 30:
                console.print(f"\n[yellow]Early stopping at epoch {epoch+1}. No improvement for {EARLY_STOPPING_PATIENCE} validations.[/yellow]")
                break
            
            if check_convergence(epoch, avg_loss, avg_sol_energy, avg_noise_energy):
                console.print(f"\n[green]Convergence achieved at epoch {epoch+1}[/green]")
                if val_acc is None:
                    val_acc, size_accuracies = validate_model(
                        backbone, energy_fn, solver, val_loader, device
                    )
                console.print(f"Final validation accuracy: {val_acc:.2%}")
                break
    
    if best_val_acc > 0:
        backbone.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        console.print(f"[green]Loaded best model (val_acc: {best_val_acc:.2%})[/green]")
    
    torch.save(backbone.state_dict(), FINAL_MODEL_PATH)
    
    console.print("\n[bold cyan]Running final test on entire dataset...[/bold cyan]")
    test_all_puzzles(backbone, energy_fn, solver, device, full_dataset)


def test_all_puzzles(model, energy_fn, solver, device, dataset):
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    size_results = {
        size_key: {"correct": 0, "total": 0, "failed_ids": []}
        for size_key in SIZE_CATEGORIES
    }
    size_results["other"] = {"correct": 0, "total": 0, "failed_ids": []}
    
    failed_examples = []
    
    model.eval()
    with torch.no_grad(), Progress() as progress:
        task = progress.add_task("[cyan]Testing all puzzles...", total=len(dataset))
        
        for batch in test_loader:
            hints = batch["hints"].to(device)
            target_grid = batch["target_grid"].to(device).float()
            puzzle_id = batch["id"][0]
            
            row_hints = hints[0, 0]
            col_hints = hints[0, 1]
            
            rows_with_hints = torch.where(row_hints.sum(dim=1) > 0)[0]
            cols_with_hints = torch.where(col_hints.sum(dim=1) > 0)[0]
            
            if len(rows_with_hints) > 0 and len(cols_with_hints) > 0:
                size = max(rows_with_hints[-1].item(), cols_with_hints[-1].item()) + 1
            else:
                size = 12
            
            size_key = "5x5"
            
            size_results[size_key]["total"] += 1
            
            init_state = torch.randn_like(target_grid) * INITIAL_NOISE_SCALE
            energy_fn.set_context(hints)
            resolved = solver.resolve(init_state, hints)
            pred_grid = (torch.sigmoid(resolved * SIGMOID_SCALE) > 0.5).float()
            
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
    
    for size_key in SIZE_CATEGORIES:
        stats = size_results[size_key]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            
            console.print(f"\n{size_key}: [bold]{stats['correct']}/{stats['total']}[/] ({acc:.2%})")
            
            if stats["correct"] < stats["total"]:
                failed_count = stats["total"] - stats["correct"]
                console.print(f"  [dim]Failed: {failed_count} puzzles[/dim]")
                if stats["failed_ids"]:
                    console.print(f"  [dim]Example failed IDs: {', '.join(stats['failed_ids'][:3])}{'...' if len(stats['failed_ids']) > 3 else ''}[/dim]")
            
            total_correct += stats["correct"]
            total_puzzles += stats["total"]
    
    overall_acc = total_correct / total_puzzles if total_puzzles > 0 else 0.0
    console.print("\n" + "-"*60)
    console.print(f"[bold]OVERALL: {total_correct}/{total_puzzles} ({overall_acc:.2%})[/bold]")
    
    if overall_acc < LOW_ACCURACY_THRESHOLD:
        console.print("[yellow]⚠️  Model accuracy is below 80%. Consider:[/yellow]")
        console.print("  - Increasing training epochs")
        console.print("  - Adjusting learning rate")
        console.print("  - Adding more data augmentation")
        console.print("  - Tuning the energy function")
    elif overall_acc < HIGH_ACCURACY_THRESHOLD:
        console.print("[yellow]⚠️  Model accuracy is good but could be improved[/yellow]")
    else:
        console.print("[green]✅ Excellent model accuracy![/green]")
    
    if failed_examples:
        torch.save(failed_examples, FAILED_EXAMPLES_PATH)
        console.print(f"\n[yellow]Saved {len(failed_examples)} failed examples to {FAILED_EXAMPLES_PATH}[/yellow]")
        console.print("[dim]Use these to debug model failures[/dim]")
    
    return overall_acc