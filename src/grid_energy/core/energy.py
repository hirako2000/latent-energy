import torch
import torch.nn.functional as F
from torch import nn

# Energy calculation constants
NEURAL_ENERGY_SCALING_FACTOR = 3.0
LOGIC_ENERGY_SCALING_FACTOR = 10.0
BINARY_ENERGY_SCALING_FACTOR = 0.1

# Grid processing constants
SIGMOID_SCALE = 3.0
BINARY_THRESHOLD_SCALE = 3.0
BINARY_THRESHOLD = 0.5

# Default size constants
DEFAULT_PUZZLE_SIZE = 12
MIN_PUZZLE_SIZE = 1

# Error reduction constants
ROW_ERROR_REDUCTION = 'sum'
COL_ERROR_REDUCTION = 'sum'


class NonogramEnergy(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.current_hints = None

    def set_context(self, hints: torch.Tensor):
        self.current_hints = hints

    def calculate_puzzle_size(self, row_hints, col_hints, batch_size):
        puzzle_sizes = []
        
        for b in range(batch_size):
            row_has_hints = (row_hints[b].sum(dim=1) > 0)
            col_has_hints = (col_hints[b].sum(dim=1) > 0)
            
            row_nonzero = torch.where(row_has_hints)[0]
            col_nonzero = torch.where(col_has_hints)[0]
            
            if len(row_nonzero) > 0 and len(col_nonzero) > 0:
                size = max(row_nonzero[-1].item(), col_nonzero[-1].item()) + MIN_PUZZLE_SIZE
            else:
                size = DEFAULT_PUZZLE_SIZE
            
            puzzle_sizes.append(size)
        
        return puzzle_sizes

    def calculate_logic_energy(self, soft_grid, row_hints, col_hints, batch_size, puzzle_sizes):
        total_logic_energy = 0.0

        for b in range(batch_size):
            size = puzzle_sizes[b]
            
            row_target = row_hints[b, :size, :].sum(dim=1)
            col_target = col_hints[b, :size, :].sum(dim=1)
            
            actual_rows = soft_grid[b, 0, :size, :size].sum(dim=1)
            actual_cols = soft_grid[b, 0, :size, :size].sum(dim=0)
            
            row_error = F.mse_loss(actual_rows, row_target, reduction=ROW_ERROR_REDUCTION) / size
            col_error = F.mse_loss(actual_cols, col_target, reduction=COL_ERROR_REDUCTION) / size
            
            total_logic_energy += (row_error + col_error)
        
        return total_logic_energy

    def calculate_binary_energy(self, grid, batch_size, puzzle_sizes):
        total_binary_energy = 0.0
        
        for b in range(batch_size):
            size = puzzle_sizes[b]
            puzzle_grid = grid[b, 0, :size, :size]
            binary_energy = torch.mean(puzzle_grid ** 2)
            total_binary_energy += binary_energy
        
        return total_binary_energy

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if self.current_hints is None:
            raise ValueError("Context not set.")

        batch_size = grid.size(0)

        neural_energy = self.model(grid, self.current_hints)

        soft_grid = torch.sigmoid(grid * SIGMOID_SCALE)

        row_hints = self.current_hints[:, 0]
        col_hints = self.current_hints[:, 1]

        puzzle_sizes = self.calculate_puzzle_size(row_hints, col_hints, batch_size)
        
        logic_energy = self.calculate_logic_energy(
            soft_grid, row_hints, col_hints, batch_size, puzzle_sizes
        )
        logic_energy = (logic_energy / batch_size) * LOGIC_ENERGY_SCALING_FACTOR
        
        binary_energy = self.calculate_binary_energy(grid, batch_size, puzzle_sizes)
        binary_energy = (binary_energy / batch_size) * BINARY_ENERGY_SCALING_FACTOR

        total_energy = neural_energy + logic_energy + binary_energy

        return total_energy

    @torch.no_grad()
    def check_logic(self, grid: torch.Tensor, hints: torch.Tensor):
        binary_grid = (torch.sigmoid(grid * BINARY_THRESHOLD_SCALE) > BINARY_THRESHOLD).float()

        row_hints = hints[:, 0]
        col_hints = hints[:, 1]

        batch_size = grid.size(0)
        total_err = 0.0

        for b in range(batch_size):
            row_has_hints = (row_hints[b].sum(dim=1) > 0)
            col_has_hints = (col_hints[b].sum(dim=1) > 0)

            row_size = torch.where(row_has_hints)[0]
            col_size = torch.where(col_has_hints)[0]

            if len(row_size) > 0 and len(col_size) > 0:
                size = max(row_size[-1].item(), col_size[-1].item()) + MIN_PUZZLE_SIZE
            else:
                size = DEFAULT_PUZZLE_SIZE

            row_target = row_hints[b, :size, :].sum(dim=1).float()
            col_target = col_hints[b, :size, :].sum(dim=1).float()

            actual_rows = binary_grid[b, 0, :size, :size].sum(dim=1)
            actual_cols = binary_grid[b, 0, :size, :size].sum(dim=0)

            row_err = torch.abs(actual_rows - row_target).sum()
            col_err = torch.abs(actual_cols - col_target).sum()

            total_err += (row_err + col_err).item()

        return total_err / batch_size