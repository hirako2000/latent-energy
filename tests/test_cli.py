import pytest
import torch
from torch import nn
import numpy as np
import re
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from grid_energy import cli
from grid_energy.cli import app
from grid_energy.train import validate_model

runner = CliRunner()

def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = re.sub(r'\[/?[a-z]+\]', '', text)
    return ansi_escape.sub('', text)

class MockModel(nn.Module):
    def __init__(self, grid_size=5, hint_dim=20):
        super().__init__()
        self.grid_size = grid_size
        self.hint_dim = hint_dim
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x, h): return x.mean() * self.scale
    def predict_grid(self, h):
        return torch.zeros((h.size(0), 1, self.grid_size, self.grid_size))

@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.ids = ["test-1", "test-2"]
    ds.__len__.return_value = 2
    ds.__getitem__.side_effect = lambda i: {
        "id": ds.ids[i],
        "hints": torch.zeros((2, 5, 10)),
        "target_grid": torch.zeros((1, 5, 5))
    }
    return ds

def test_validate_model_null_accuracy():
    model = MockModel()
    energy = MagicMock()
    solver = MagicMock()
    solver.resolve.return_value = torch.zeros((1, 1, 5, 5))
    
    loader = [{
        "hints": torch.zeros((1, 2, 5, 10)),
        "target_grid": torch.zeros((1, 1, 5, 5))
    }]

    with patch("grid_energy.core.solver.KineticSolver", return_value=solver):
        with patch.object(energy, 'check_logic', return_value=None):
            acc, size_accs = validate_model(model, energy, solver, loader, "cpu")
            assert "5x5" in size_accs
            assert size_accs["5x5"] == 1.0

def test_calculate_puzzle_size_logic():
    h1 = torch.zeros((1, 2, 5, 10))
    h1[0, 0, 1, 0] = 1.0
    h1[0, 1, 3, 0] = 1.0
    assert cli.calculate_puzzle_size(h1)[0] == 4

    h2 = torch.zeros((1, 2, 5, 10))
    h2[0, 0, 2, 0] = 1.0
    assert cli.calculate_puzzle_size(h2)[0] == 3

    h3 = torch.zeros((1, 2, 5, 10))
    h3[0, 1, 0, 0] = 1.0
    assert cli.calculate_puzzle_size(h3)[0] == 1
    
    h4 = torch.zeros((1, 2, 5, 10))
    assert cli.calculate_puzzle_size(h4)[0] == 5

def test_resolve_retry_logic():
    model = MockModel()
    solver = MagicMock()
    energy = MagicMock()
    hints = torch.zeros((1, 2, 5, 10))
    solver.resolve.return_value = torch.zeros((1, 1, 5, 5))

    with patch("grid_energy.cli.get_procedural_score") as mock_score:
        mock_score.side_effect = [5, 5, 0]
        best, err, tries, ms = cli.resolve_puzzle_with_retries(
            model, solver, energy, hints, 3, "id", 5, [], []
        )
        assert tries == 3
        assert err == 0

def test_load_model_failure_handling(tmp_path):
    weights = tmp_path / "bad.pt"
    weights.touch()
    with patch("torch.load", side_effect=Exception("Corrupt")), \
         patch("grid_energy.cli.NonogramCNN", return_value=MockModel()):
        m = cli.load_model(torch.device("cpu"), weights, 20)
        assert isinstance(m, MockModel)

def test_select_puzzle_logic(mock_dataset):
    sample, p_id, idx = cli.select_puzzle(mock_dataset, "", 42)
    assert p_id == "test-1"
    
    sample, p_id, idx = cli.select_puzzle(mock_dataset, "unknown", 42)
    assert sample is None

@patch("grid_energy.cli.CroissantDataset")
@patch("grid_energy.cli.load_model")
@patch("grid_energy.cli.KineticSolver")
def test_resolve_full_flow(mock_sol, mock_load, mock_ds_cls, mock_dataset):
    mock_ds_cls.return_value = mock_dataset
    mock_load.return_value = MockModel()
    mock_sol.return_value.resolve.return_value = torch.ones((1, 1, 5, 5))
    
    result = runner.invoke(app, ["resolve", "--puzzle-id", "test-1", "--compare"])
    assert result.exit_code == 0
    
    with patch("grid_energy.cli.resolve_puzzle_with_retries", return_value=(None, 0, 0, 0)):
        result_fail = runner.invoke(app, ["resolve", "--puzzle-id", "test-1"])
        assert "Error" in strip_ansi(result_fail.output)

@patch("grid_energy.cli.CroissantDataset")
def test_list_ids_ui(mock_ds_cls):
    ds = MagicMock()
    ds.ids = ["id_" + "x" * 100]
    ds.__len__.return_value = 1
    mock_ds_cls.return_value = ds
    result = runner.invoke(app, ["list-ids", "--limit", "1"])
    assert "..." in result.output
    
    mock_ds_cls.side_effect = Exception("Fail")
    result_err = runner.invoke(app, ["list-ids"])
    assert "Error" in strip_ansi(result_err.output)

@patch("grid_energy.cli.train_ebm")
def test_train_cli(mock_train):
    mock_train.side_effect = RuntimeError("Error")
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "Halted" in strip_ansi(result.output)

def test_ingest_routing():
    with patch("grid_energy.cli.fetch_bronze_data") as b, \
         patch("grid_energy.cli.process_silver_data") as s, \
         patch("grid_energy.cli.synthesize_gold") as g:
        runner.invoke(app, ["ingest", "bronze"])
        runner.invoke(app, ["ingest", "silver"])
        runner.invoke(app, ["ingest", "gold"])
        assert b.called and s.called and g.called
    
    res = runner.invoke(app, ["ingest", "unknown"])
    assert "Error" in res.output

def test_utility_calls():
    with patch("grid_energy.cli.create_multi_cell_atlas") as v:
        runner.invoke(app, ["visualize-energy"])
        assert v.called
    
    with patch("grid_energy.cli.app") as m:
        cli.main()
        assert m.called

def test_device_selection():
    with patch("torch.cuda.is_available", return_value=True):
        assert cli.get_device().type == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        assert cli.get_device().type == "cpu"

def test_display_and_hints():
    cli.display_solution(np.ones((5, 5)), 5)
    rows = np.array([[1, 2, 0], [3, 0, 0]])
    cols = np.array([[1, 0, 0], [0, 0, 0]])
    r, c = cli.extract_clean_hints(rows, cols, 2)
    assert r == [[1, 2], [3]]
    assert c == [[1], []]

@patch("grid_energy.cli.CroissantDataset")
@patch("grid_energy.cli.load_model")
@patch("grid_energy.cli.KineticSolver")
def test_resolve_status_reporting(mock_sol_cls, mock_load, mock_ds_cls, mock_dataset):
    mock_ds_cls.return_value = mock_dataset
    mock_load.return_value = MockModel()
    mock_sol_cls.return_value.resolve.return_value = torch.ones((1, 1, 5, 5))
    
    with patch("grid_energy.cli.get_procedural_score", return_value=5):
        res = runner.invoke(app, ["resolve", "--puzzle-id", "test-1", "--max-retries", "1"])
        output = strip_ansi(res.output)
        assert "FAILED" in output
        assert "5 lines wrong" in output