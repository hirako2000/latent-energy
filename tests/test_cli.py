import pytest
import torch
import re
from unittest.mock import MagicMock, patch
from pathlib import Path
from typer.testing import CliRunner

from grid_energy import cli
from grid_energy.cli import app

runner = CliRunner()

def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = re.sub(r'\[/?[a-z]+\]', '', text)
    return ansi_escape.sub('', text)

@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.ids = ["test-1", "test-2"]
    ds.__len__.return_value = 2
    ds.__getitem__.side_effect = lambda i: {
        "id": ds.ids[i] if i < len(ds.ids) else "out-of-bounds",
        "hints": torch.zeros((2, 5, 5)),
        "target_grid": torch.zeros((1, 5, 5))
    }
    return ds

def test_load_model_exception_handling(tmp_path: Path):
    weights = tmp_path / "corrupt.pt"
    weights.touch()
    with patch("torch.load", side_effect=RuntimeError("Mocked Load Failure")), \
         patch("grid_energy.cli.NonogramCNN"):
        model = cli.load_model(torch.device("cpu"), weights, 10)
        assert model is not None

def test_get_device_cuda_logic():
    with patch("torch.cuda.is_available", return_value=True):
        device = cli.get_device()
        assert device.type == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        device = cli.get_device()
        assert device.type == "cpu"

def test_cuda_synchronize_logic():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize") as mock_sync:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mock_sync.assert_called_once()

def test_select_puzzle_random_logic(mock_dataset):
    with patch("random.randint", return_value=1):
        sample, p_id, idx = cli.select_puzzle(mock_dataset, "", 123)
        assert p_id == "test-2"
        assert idx == 1

def test_select_puzzle_value_error(mock_dataset):
    mock_dataset.ids = MagicMock(spec=list)
    mock_dataset.ids.index.side_effect = ValueError()
    sample, p_id, idx = cli.select_puzzle(mock_dataset, "missing", 0)
    assert sample is None

def test_calculate_puzzle_size_branches():
    hints_rows = torch.zeros((1, 2, 5, 5))
    hints_rows[0, 0, 2, 0] = 1
    size_r, _, _ = cli.calculate_puzzle_size(hints_rows)
    assert size_r == 3

    hints_cols = torch.zeros((1, 2, 5, 5))
    hints_cols[0, 1, 3, 0] = 1
    size_c, _, _ = cli.calculate_puzzle_size(hints_cols)
    assert size_c == 4

@patch("grid_energy.cli.process_silver_data")
def test_ingest_silver_tier(mock_silver):
    result = runner.invoke(app, ["ingest", "silver"])
    assert result.exit_code == 0
    mock_silver.assert_called_once()

@patch("grid_energy.cli.synthesize_gold")
def test_ingest_gold_tier(mock_gold):
    result = runner.invoke(app, ["ingest", "gold"])
    assert result.exit_code == 0
    mock_gold.assert_called_once()

def test_ingest_unknown_tier():
    result = runner.invoke(app, ["ingest", "invalid"])
    assert "Unknown tier" in strip_ansi(result.output)

@patch("grid_energy.cli.CroissantDataset")
def test_list_ids_success_and_truncation(mock_ds_class):
    mock_ds = MagicMock()
    long_id = "a" * 60
    mock_ds.ids = [long_id, "short-id"]
    mock_ds.__len__.return_value = 2
    mock_ds_class.return_value = mock_ds

    result = runner.invoke(app, ["list-ids", "--limit", "2"])
    clean_output = strip_ansi(result.output)
    assert "Available IDs" in clean_output
    assert "aaaaa..." in clean_output

@patch("grid_energy.cli.CroissantDataset", side_effect=Exception("Load Fail"))
def test_list_ids_exception(mock_ds_class):
    result = runner.invoke(app, ["list-ids"])
    assert "Error" in strip_ansi(result.output)
    assert "Load Fail" in strip_ansi(result.output)

@patch("grid_energy.cli.select_puzzle", return_value=(None, None, None))
def test_resolve_early_exit_on_none_sample(mock_select):
    result = runner.invoke(app, ["resolve", "--puzzle-id", "invalid"])
    assert result.exit_code == 0

@patch("grid_energy.cli.CroissantDataset")
@patch("grid_energy.cli.load_model")
@patch("grid_energy.cli.KineticSolver")
def test_resolve_failed_to_resolve_puzzle(mock_solver_cls, mock_load, mock_ds_cls, mock_dataset):
    mock_ds_cls.return_value = mock_dataset
    mock_solver_cls.return_value.resolve.return_value = None
    result = runner.invoke(app, ["resolve", "--puzzle-id", "test-1"])
    assert result.exit_code == 1

@patch("grid_energy.cli.train_ebm", side_effect=Exception("Critical Failure"))
def test_train_engine_halt(mock_train):
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "Halted" in strip_ansi(result.output)

def test_generate_seed_from_puzzle_id():
    seed = cli.generate_seed_from_puzzle_id("abc", 0)
    assert isinstance(seed, int)

def test_display_solution_rendering():
    grid = torch.tensor([[1, 0], [0, 1]])
    cli.display_solution(grid, 2)

def test_main_entrypoint():
    with patch("grid_energy.cli.app") as mock_app:
        cli.main()
        mock_app.assert_called_once()