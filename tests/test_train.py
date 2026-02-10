import torch
from torch import nn
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from grid_energy.train import (
    create_optimization_components,
    create_stats_table,
    generate_hard_negative,
    compute_training_loss,
    train_epoch,
    format_accuracy_value,
    add_table_row,
    check_convergence,
    validate_model,
    pretrain_supervised,
    train_ebm,
    check_all_puzzles
)

class MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(1, requires_grad=True))
    
    def predict_grid(self, hints: torch.Tensor) -> torch.Tensor:
        return torch.randn(hints.size(0), 1, 5, 5) * self.param

@pytest.fixture
def mock_model() -> MockModel:
    return MockModel()

@pytest.fixture
def mock_energy_fn() -> MagicMock:
    fn = MagicMock()
    fn.return_value = torch.tensor([1.0], requires_grad=True)
    fn.check_logic.return_value = 0.1
    return fn

@pytest.fixture
def sample_batch() -> Dict[str, Any]:
    return {
        "target_grid": torch.ones((2, 1, 5, 5)),
        "hints": torch.zeros((2, 2, 5, 10)),
        "id": ["p1", "p2"]
    }

def test_create_optimization_components(mock_model):
    opt, sched = create_optimization_components(mock_model, 1e-4)
    assert isinstance(opt, torch.optim.AdamW)

def test_create_stats_table():
    table = create_stats_table("cpu")
    assert table.title is not None

def test_generate_hard_negative():
    pos = torch.randn(1, 1, 5, 5)
    neg = generate_hard_negative(pos, None)
    assert neg.shape == pos.shape
    assert not torch.equal(pos, neg)

def test_compute_training_loss(mock_energy_fn):
    pos = torch.randn(2, 1, 5, 5)
    loss, e_sol, e_noise, e_wrong = compute_training_loss(mock_energy_fn, pos, pos, pos)
    assert isinstance(loss, torch.Tensor)

def test_train_epoch(mock_model, mock_energy_fn, sample_batch):
    optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)
    res = train_epoch(mock_model, mock_energy_fn, [sample_batch], optimizer, "cpu")
    assert len(res) == 5

def test_format_accuracy_value():
    assert "green" in format_accuracy_value(0.96)
    assert "yellow" in format_accuracy_value(0.85)
    assert "red" in format_accuracy_value(0.50)

def test_add_table_row_branches():
    table = MagicMock()
    add_table_row(table, 0, 0.1, 0.1, 0.1, 0.1, 0.9, {"5x5": 0.9}, 1e-4)
    add_table_row(table, 1, 0.1, 0.1, 0.1, 0.1, 0.8, {}, 1e-4)
    add_table_row(table, 2, 0.1, 0.1, 0.1, 0.1, None, {}, 1e-4)
    assert table.add_row.call_count == 3

def test_check_convergence_branches():
    assert not check_convergence(5, 0.001, 0.1, 2.0, 2.0)
    assert check_convergence(25, 0.001, 0.1, 2.0, 2.0) is True
    assert not check_convergence(25, 0.5, 0.1, 2.0, 2.0)

def test_validate_model_logic(mock_model, mock_energy_fn):
    solver = MagicMock()
    solver.resolve.return_value = torch.ones((1, 1, 5, 5))
    loader = [{"target_grid": torch.ones((1, 1, 5, 5)), "hints": torch.zeros((1, 2, 5, 10))}]
    acc, size_accs = validate_model(mock_model, mock_energy_fn, solver, loader, "cpu")
    assert acc == 1.0

def test_validate_model_with_hints_fallback(mock_model, mock_energy_fn):
    solver = MagicMock()
    solver.resolve.return_value = torch.ones((1, 1, 5, 5))
    hints = torch.zeros((1, 2, 5, 10))
    hints[0, 0, 2, 0] = 1.0
    loader = [{"target_grid": torch.ones((1, 1, 5, 5)), "hints": hints}]
    acc, _ = validate_model(mock_model, mock_energy_fn, solver, loader, "cpu")
    assert acc == 1.0

def test_pretrain_supervised_execution(mock_model):
    loader = [{"target_grid": torch.ones((1, 1, 5, 5)), "hints": torch.zeros((1, 2, 5, 10))}]
    with patch("torch.save"), patch("torch.load", return_value=mock_model.state_dict()), patch("grid_energy.train.console"):
        model = pretrain_supervised(mock_model, loader, loader, "cpu", epochs=1)
        assert model is not None

@patch("grid_energy.train.CroissantDataset")
@patch("grid_energy.train.DataLoader")
@patch("grid_energy.train.train_test_split")
@patch("grid_energy.train.NonogramCNN")
@patch("grid_energy.train.check_all_puzzles")
@patch("torch.save")
@patch("torch.load")
def test_train_ebm_complex_flow(mock_load, mock_save, mock_check, mock_cnn, mock_split, mock_dl, mock_ds, mock_model):
    mock_split.return_value = ([0], [0])
    mock_cnn.return_value = mock_model
    mock_load.return_value = mock_model.state_dict()
    mock_ds.return_value = MagicMock(__len__=lambda x: 1)
    
    with patch("grid_energy.train.train_epoch", return_value=(0.1, 0.1, 0.1, 0.1, 0.1)), \
         patch("grid_energy.train.Live"):
        with patch("grid_energy.train.validate_model", side_effect=[(0.5, {"5x5": 0.5})] * 100):
            train_ebm(epochs=2, validation_freq=1)

    with patch("grid_energy.train.train_epoch", return_value=(0.001, 0.1, 2.0, 2.0, 0.1)), \
         patch("grid_energy.train.Live"):
        with patch("grid_energy.train.validate_model", side_effect=[(1.0, {"5x5": 1.0})] * 100):
            train_ebm(epochs=21, validation_freq=100)

def test_check_all_puzzles_full_coverage(mock_model, mock_energy_fn):
    solver = MagicMock()
    solver.resolve.return_value = torch.zeros((1, 1, 5, 5))
    loader = [{"target_grid": torch.ones((1, 1, 5, 5)), "hints": torch.zeros((1, 2, 5, 10)), "id": ["f1"]}]
    mock_ds = MagicMock(__len__=lambda x: 1)
    
    with patch("grid_energy.train.DataLoader", return_value=loader), patch("torch.save"), patch("grid_energy.train.console"):
        acc_low = check_all_puzzles(mock_model, mock_energy_fn, solver, "cpu", mock_ds)
        assert acc_low == 0.0

    solver.resolve.return_value = torch.ones((1, 1, 5, 5))
    with patch("grid_energy.train.DataLoader", return_value=loader), patch("torch.save"), patch("grid_energy.train.console"):
        acc_high = check_all_puzzles(mock_model, mock_energy_fn, solver, "cpu", mock_ds)
        assert acc_high == 1.0
        
    with patch("grid_energy.train.DataLoader", return_value=loader), patch("torch.save"), patch("grid_energy.train.console"):
        with patch("grid_energy.train.HIGH_ACCURACY_THRESHOLD", 1.1), patch("grid_energy.train.LOW_ACCURACY_THRESHOLD", -0.1):
            check_all_puzzles(mock_model, mock_energy_fn, solver, "cpu", mock_ds)