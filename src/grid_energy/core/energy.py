import torch
import torch.nn as nn
import torch.nn.functional as F

class NonogramEnergy(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.current_hints = None

    def set_context(self, hints: torch.Tensor):
        self.current_hints = hints

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if self.current_hints is None:
            raise ValueError("Context not set.")
        
        neural_energy = self.model(grid, self.current_hints)
        
        soft_grid = torch.sigmoid(grid * 3.0)
        
        row_hints = self.current_hints[:, 0]
        col_hints = self.current_hints[:, 1]
        
        row_target = row_hints.sum(dim=2)
        col_target = col_hints.sum(dim=2)
        
        actual_rows = soft_grid.sum(dim=3).squeeze(1)
        actual_cols = soft_grid.sum(dim=2).squeeze(1)
        
        row_error = torch.abs(actual_rows - row_target).sum(dim=1)
        col_error = torch.abs(actual_cols - col_target).sum(dim=1)
        
        logic_energy = (row_error + col_error) / (self.current_hints.shape[2] * 2)
        
        binary_energy = torch.mean(torch.sigmoid(10 * (1.0 - grid.abs())), dim=(1, 2, 3))
        
        total_energy = neural_energy + logic_energy * 100.0 + binary_energy * 0.1
        
        return total_energy

    @torch.no_grad()
    def check_logic(self, grid: torch.Tensor, hints: torch.Tensor):
        binary_grid = (torch.sigmoid(grid * 3.0) > 0.5).float()
        
        row_hints = hints[:, 0]
        col_hints = hints[:, 1]
        
        row_target = row_hints.sum(dim=2).float()
        col_target = col_hints.sum(dim=2).float()
        
        actual_rows = binary_grid.sum(dim=3).squeeze(1)
        actual_cols = binary_grid.sum(dim=2).squeeze(1)
        
        row_err = torch.abs(actual_rows - row_target).sum(dim=1)
        col_err = torch.abs(actual_cols - col_target).sum(dim=1)
        
        return (row_err + col_err).mean().item()