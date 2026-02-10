import torch
from torch import nn
import pytest
from unittest.mock import patch
from grid_energy.core.energy import NonogramEnergy

class MockBackbone(nn.Module):
    def forward(self, grid: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        return torch.mean(grid) + torch.mean(hints)

@pytest.fixture
def mock_model() -> MockBackbone:
    return MockBackbone()

@pytest.fixture
def energy_fn(mock_model: MockBackbone) -> NonogramEnergy:
    return NonogramEnergy(mock_model)

@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.randn(2, 1, 5, 5)
    hints = torch.zeros(2, 2, 5, 10)
    
    hints[0, 0, 0, 0] = 1.0
    hints[0, 0, 1, 0] = 2.0
    hints[0, 1, 0, 0] = 1.0
    hints[0, 1, 1, 0] = 2.0
    
    hints[1, 0, 0, 0] = 3.0
    hints[1, 1, 0, 0] = 3.0
    
    return grid, hints

def test_set_context(energy_fn: NonogramEnergy) -> None:
    hints = torch.randn(1, 2, 5, 10)
    energy_fn.set_context(hints)
    assert energy_fn.current_hints is hints

def test_forward_no_context(energy_fn: NonogramEnergy) -> None:
    grid = torch.randn(1, 1, 5, 5)
    with pytest.raises(ValueError, match="Context not set."):
        energy_fn(grid)

def test_calculate_puzzle_size_logic(energy_fn: NonogramEnergy) -> None:
    row_hints = torch.zeros(2, 5, 10)
    col_hints = torch.zeros(2, 5, 10)
    
    row_hints[0, 2, 0] = 1.0
    col_hints[0, 1, 0] = 1.0
    
    sizes = energy_fn.calculate_puzzle_size(row_hints, col_hints, 2)
    assert sizes[0] == 3
    assert sizes[1] == 5

def test_forward_execution(energy_fn: NonogramEnergy, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    grid, hints = sample_data
    energy_fn.set_context(hints)
    energy = energy_fn(grid)
    
    assert isinstance(energy, torch.Tensor)
    assert energy.numel() == 1

def test_check_logic_accuracy(energy_fn: NonogramEnergy) -> None:
    hints = torch.zeros(1, 2, 5, 10)
    hints[0, 0, 0, 0] = 2.0
    hints[0, 1, 0, 0] = 2.0
    
    perfect_grid = torch.zeros(1, 1, 5, 5)
    perfect_grid[0, 0, 0, 0] = 10.0
    perfect_grid[0, 0, 0, 1] = 10.0
    
    error = energy_fn.check_logic(perfect_grid, hints)
    assert float(error) == 2.0
    
    wrong_grid = torch.zeros(1, 1, 5, 5)
    error_val = energy_fn.check_logic(wrong_grid, hints)
    assert float(error_val) > 0.0

def test_binary_energy_calculation(energy_fn: NonogramEnergy) -> None:
    grid = torch.ones(1, 1, 5, 5) * 2.0
    sizes = [5]
    energy = energy_fn.calculate_binary_energy(grid, 1, sizes)
    assert isinstance(energy, torch.Tensor)
    assert torch.allclose(energy, torch.tensor(4.0))

def test_logic_energy_reduction(energy_fn: NonogramEnergy) -> None:
    soft_grid = torch.ones(1, 1, 5, 5)
    row_hints = torch.zeros(1, 5, 10)
    col_hints = torch.zeros(1, 5, 10)
    row_hints[0, 0, 0] = 5.0
    col_hints[0, 0, 0] = 5.0
    
    sizes = [5]
    energy = energy_fn.calculate_logic_energy(soft_grid, row_hints, col_hints, 1, sizes)
    assert float(energy) >= 0.0

def test_check_logic_fallback_size(energy_fn: NonogramEnergy) -> None:
    grid = torch.zeros(1, 1, 5, 5)
    hints = torch.zeros(1, 2, 5, 10)
    error = energy_fn.check_logic(grid, hints)
    assert isinstance(error, float)

def test_neural_energy_contribution(energy_fn: NonogramEnergy, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    grid, hints = sample_data
    energy_fn.set_context(hints)
    
    with patch.object(MockBackbone, 'forward', return_value=torch.tensor(5.0)) as mock_fwd:
        energy = energy_fn(grid)
        assert mock_fwd.called
        assert float(energy) > 0