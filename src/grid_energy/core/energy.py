import torch
import torch.nn.functional as F
from torch import nn


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

        batch_size = grid.size(0)

        neural_energy = self.model(grid, self.current_hints)

        soft_grid = torch.sigmoid(grid * 3.0)

        row_hints = self.current_hints[:, 0]  # [B, 12, max_hint_len]
        col_hints = self.current_hints[:, 1]  # [B, 12, max_hint_len]

        # Find actual puzzle size for each sample
        # lets count rows/cols with non-zero hints
        row_has_hints = (row_hints.sum(dim=2) > 0).float()  # [B, 12]
        col_has_hints = (col_hints.sum(dim=2) > 0).float()  # [B, 12]

        # size is max of row/col with hints
        puzzle_sizes = []
        for b in range(batch_size):
            # last row/col with hints
            row_size = torch.where(row_has_hints[b] > 0)[0]
            col_size = torch.where(col_has_hints[b] > 0)[0]

            if len(row_size) > 0 and len(col_size) > 0:
                size = max(row_size[-1].item(), col_size[-1].item()) + 1
            else:
                size = 12  # Default to max

            puzzle_sizes.append(size)

        total_logic_energy = 0.0

        for b in range(batch_size):
            size = puzzle_sizes[b]

            # relevant hints for this puzzle
            row_target = row_hints[b, :size, :].sum(dim=1)  # [size]
            col_target = col_hints[b, :size, :].sum(dim=1)  # [size]

            # relevant grid area
            actual_rows = soft_grid[b, 0, :size, :size].sum(dim=1)  # [size]
            actual_cols = soft_grid[b, 0, :size, :size].sum(dim=0)  # [size]

            # error only on actual puzzle area
            row_error = F.mse_loss(actual_rows, row_target, reduction='sum') / size
            col_error = F.mse_loss(actual_cols, col_target, reduction='sum') / size

            total_logic_energy += (row_error + col_error)

        logic_energy = (total_logic_energy / batch_size) * 10.0

        # Binary regularization (only on actual puzzle area)
        total_binary_energy = 0.0
        for b in range(batch_size):
            size = puzzle_sizes[b]
            puzzle_grid = grid[b, 0, :size, :size]
            binary_energy = torch.mean(puzzle_grid ** 2)
            total_binary_energy += binary_energy

        binary_energy = (total_binary_energy / batch_size) * 0.1

        total_energy = neural_energy + logic_energy + binary_energy

        return total_energy

    @torch.no_grad()
    def check_logic(self, grid: torch.Tensor, hints: torch.Tensor):
        binary_grid = (torch.sigmoid(grid * 3.0) > 0.5).float()

        row_hints = hints[:, 0]
        col_hints = hints[:, 1]

        batch_size = grid.size(0)
        total_err = 0.0

        for b in range(batch_size):
            # puzzle size
            row_has_hints = (row_hints[b].sum(dim=1) > 0)
            col_has_hints = (col_hints[b].sum(dim=1) > 0)

            row_size = torch.where(row_has_hints)[0]
            col_size = torch.where(col_has_hints)[0]

            if len(row_size) > 0 and len(col_size) > 0:
                size = max(row_size[-1].item(), col_size[-1].item()) + 1
            else:
                size = 12

            # only actual puzzle area
            row_target = row_hints[b, :size, :].sum(dim=1).float()
            col_target = col_hints[b, :size, :].sum(dim=1).float()

            actual_rows = binary_grid[b, 0, :size, :size].sum(dim=1)
            actual_cols = binary_grid[b, 0, :size, :size].sum(dim=0)

            row_err = torch.abs(actual_rows - row_target).sum()
            col_err = torch.abs(actual_cols - col_target).sum()

            total_err += (row_err + col_err).item()

        return total_err / batch_size
