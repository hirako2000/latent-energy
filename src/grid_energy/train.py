import torch
import torch.optim as optim
import torch.nn.functional as F
from rich.live import Live
from rich.table import Table

def train_ebm(epochs: int, batch_size: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    from grid_energy.core.models import NonogramCNN
    from grid_energy.core.energy import NonogramEnergy
    from grid_energy.core.solver import KineticSolver
    from grid_energy.data.loader import get_dataloader

    backbone = NonogramCNN(grid_size=12).to(device)
    energy_fn = NonogramEnergy(backbone).to(device)
    
    optimizer = optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-5)
    
    loader = get_dataloader(batch_size=batch_size, shuffle=True)

    stats_table = Table(title=f"EBM Training ({device})")
    stats_table.add_column("Epoch", justify="center")
    stats_table.add_column("Loss", justify="right", style="cyan")
    stats_table.add_column("Solution Energy", justify="right", style="green")
    stats_table.add_column("Noise Energy", justify="right", style="magenta")
    stats_table.add_column("LogicErr", justify="right", style="red")

    with Live(stats_table, refresh_per_second=2):
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_sol_energy = 0.0
            epoch_noise_energy = 0.0
            epoch_logic = 0.0
            total_batches = 0
            
            backbone.train()
            
            for batch in loader:
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
            
            stats_table.add_row(
                f"{epoch+1:02d}",
                f"{avg_loss:.3f}",
                f"{avg_sol_energy:.3f}",
                f"{avg_noise_energy:.3f}",
                f"{avg_logic:.3f}"
            )
            
            if epoch > 10 and avg_loss < 0.1 and avg_sol_energy < avg_noise_energy:
                print(f"\nGood separation. Stopping early.")
                break
    
    torch.save(backbone.state_dict(), "data/models/ebm_trained.pt")
    print(f"Training complete.")