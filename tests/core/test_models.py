import torch
import pytest
from torch import nn
from grid_energy.core.models import NonogramCNN

@pytest.fixture
def model() -> NonogramCNN:
    return NonogramCNN(grid_size=5, hint_dim=20)

@pytest.fixture
def sample_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 4
    grid = torch.randn(batch_size, 1, 5, 5)
    hints = torch.randn(batch_size, 2, 5, 2) # 2 * 5 * 2 = 20
    return grid, hints

def test_model_initialization(model: NonogramCNN) -> None:
    assert model.grid_size == 5
    assert model.hint_dim == 20
    assert isinstance(model.scale, nn.Parameter)
    assert model.scale.item() == 1.0

def test_forward_output_shape(model: NonogramCNN, sample_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    grid, hints = sample_inputs
    output = model(grid, hints)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (grid.size(0),)

def test_hint_dimension_mismatch(model: NonogramCNN) -> None:
    grid = torch.randn(1, 1, 5, 5)
    invalid_hints = torch.randn(1, 1, 1, 1) # Dim 1 instead of 20 to test
    
    with pytest.raises(ValueError, match="Hint dimension mismatch"):
        model(grid, invalid_hints)

def test_predict_grid_values(model: NonogramCNN) -> None:
    batch_size = 2
    hints = torch.randn(batch_size, 2, 5, 2)
    grid_pred = model.predict_grid(hints)
    
    assert grid_pred.shape == (batch_size, 1, 5, 5)
    # output must be in range [-1, 1] due to (sigm - 0.5) * 2 logic
    assert torch.all(grid_pred >= -1.0)
    assert torch.all(grid_pred <= 1.0)

def test_parameter_gradient_flow(model: NonogramCNN, sample_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    grid, hints = sample_inputs
    output = model(grid, hints)
    loss = output.sum()
    loss.backward()
    
    # gradients
    assert model.scale.grad is not None
    assert model.conv1.weight.grad is not None
    assert model.hint_encoder[0].weight.grad is not None # type: ignore

def test_spatial_consistency(model: NonogramCNN) -> None:
    # even with batch size 1, the view operations should succeed
    grid = torch.randn(1, 1, 5, 5)
    hints = torch.randn(1, 20)
    output = model(grid, hints)
    assert output.shape == (1,)

def test_predict_grid_determinism(model: NonogramCNN) -> None:
    hints = torch.randn(1, 2, 5, 2)
    pred1 = model.predict_grid(hints)
    pred2 = model.predict_grid(hints)
    assert torch.equal(pred1, pred2)