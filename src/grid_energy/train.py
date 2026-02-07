import torch
import torch.optim as optim
import torch.nn.functional as F
from rich.live import Live
from rich.table import Table
from sklearn.metrics import f1_score

def train_ebm(epochs: int, batch_size: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    from grid_energy.core.models import NonogramCNN
    from grid_energy.core.energy import NonogramEnergy
    from grid_energy.core.solver import KineticSolver
    from grid_energy.data.loader import get_dataloader

    backbone = NonogramCNN(grid_size=12).to(device)
    energy_fn = NonogramEnergy(backbone).to(device)
    
    train_solver = KineticSolver(energy_fn, step_size=0.02, n_steps=30)
    eval_solver = KineticSolver(energy_fn, step_size=0.01, n_steps=80)
    
    optimizer = optim.Adam(backbone.parameters(), lr=lr)
    
    loader = get_dataloader(batch_size=batch_size, shuffle=True)

    stats_table = Table(title=f"EBM Training ({device})")
    stats_table.add_column("Epoch", justify="center")
    stats_table.add_column("Loss", justify="right", style="cyan")
    stats_table.add_column("F1", justify="right", style="green")
    stats_table.add_column("LogicErr", justify="right", style="red")

    with Live(stats_table, refresh_per_second=2):
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_logic = 0.0
            total_batches = 0
            
            backbone.train()
            
            for batch in loader:
                pos_grid = batch["target_grid"].to(device).float()
                hints = batch["hints"].to(device)
                
                pos_continuous = (pos_grid - 0.5) * 2.0
                
                energy_fn.set_context(hints)
                
                optimizer.zero_grad()
                
                e_pos = energy_fn(pos_continuous)
                
                backbone.eval()
                with torch.no_grad():
                    neg_start = torch.randn_like(pos_continuous) * 0.05
                    neg_grid = train_solver.resolve(neg_start, hints)
                
                backbone.train()
                e_neg = energy_fn(neg_grid)
                
                loss = F.relu(e_pos - e_neg + 0.5).mean()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 0.5)
                optimizer.step()
                
                logic_err = energy_fn.check_logic(neg_grid, hints)
                
                epoch_loss += loss.item()
                epoch_logic += logic_err
                total_batches += 1
            
            backbone.eval()
            with torch.no_grad():
                eval_batch = next(iter(loader))
                eval_grid = eval_batch["target_grid"].to(device).float()
                eval_hints = eval_batch["hints"].to(device)
                
                energy_fn.set_context(eval_hints)
                
                eval_start = torch.randn_like(eval_grid) * 0.05
                resolved = eval_solver.resolve(eval_start, eval_hints)
                
                binary_res = (torch.sigmoid(resolved * 3.0) > 0.5).float()
                binary_target = eval_grid
                
                f1 = f1_score(
                    binary_target.cpu().numpy().flatten(),
                    binary_res.cpu().numpy().flatten(),
                    zero_division=0
                )
                
                avg_loss = epoch_loss / total_batches
                avg_logic = epoch_logic / total_batches
                
                stats_table.add_row(
                    f"{epoch+1:02d}",
                    f"{avg_loss:.3f}",
                    f"{f1:.4f}",
                    f"{avg_logic:.3f}"
                )
    
    torch.save(backbone.state_dict(), "data/models/ebm_final.pt")
    print(f"Training complete.")