import pytest
import torch
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
import grid_energy.cli as cli
from grid_energy.cli import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    mock_ds = MagicMock()
    mock_ds.ids = ["puz_001", "puz_002"]
    mock_ds.__len__.return_value = 2
    hints = torch.zeros((2, 12, 12))
    hints[0, 0, 0] = 1.0 
    hints[1, 0, 0] = 1.0
    mock_ds.__getitem__.return_value = {
        "id": "puz_001",
        "hints": hints,
        "target_grid": torch.zeros((1, 1, 12, 12))
    }
    mocker.patch("grid_energy.cli.CroissantDataset", return_value=mock_ds)
    cli.ds = mock_ds 
    mocker.patch("grid_energy.cli.settings")
    mocker.patch("grid_energy.cli.torch.load")
    mocker.patch("grid_energy.cli.torch.manual_seed")
    mocker.patch("grid_energy.cli.torch.cuda.is_available", return_value=False)

@pytest.mark.parametrize("tier", ["bronze", "silver", "gold", "unknown"])
def test_ingest(mocker, tier):
    mocker.patch("grid_energy.cli.fetch_bronze_data")
    mocker.patch("grid_energy.cli.process_silver_data")
    mocker.patch("grid_energy.cli.synthesize_gold")
    result = runner.invoke(app, ["ingest", tier])
    assert result.exit_code == 0

def test_train(mocker):
    mock_train = mocker.patch("grid_energy.train.train_ebm")
    runner.invoke(app, ["train"])
    mock_train.side_effect = Exception("OOM")
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1

def test_resolve(mocker):
    mocker.patch("grid_energy.cli.NonogramCNN")
    mocker.patch("grid_energy.cli.NonogramEnergy")
    mock_solver = mocker.patch("grid_energy.cli.KineticSolver").return_value
    solved_state = torch.full((1, 1, 12, 12), 10.0)
    mock_solver.resolve.return_value = solved_state
    
    with mocker.patch("pathlib.Path.exists", side_effect=[True, False]):
        runner.invoke(app, ["resolve", "--puzzle-id", ""])
        mock_solver.resolve.side_effect = [torch.zeros((1, 1, 12, 12)), solved_state]
        cli.ds.ids = ["puz_001"]
        result = runner.invoke(app, ["resolve", "--puzzle-id", "puz_001", "--max-retries", "2"])
        assert "SOLVED" in result.stdout

def test_resolve_errors(mocker):
    cli.ds.ids = MagicMock()
    cli.ds.ids.index.side_effect = ValueError()
    result = runner.invoke(app, ["resolve", "--puzzle-id", "missing"])
    assert "not found" in result.stdout

def test_diagnose(mocker):
    mocker.patch("grid_energy.cli.NonogramCNN")
    mock_en = mocker.patch("grid_energy.cli.NonogramEnergy").return_value
    cli.ds.ids = ["puz_001"]
    mocker.patch("pathlib.Path.exists", return_value=True)
    
    mock_en.side_effect = [torch.tensor(0.1), torch.tensor(0.9)]
    runner.invoke(app, ["diagnose", "puz_001"])

    mock_en.side_effect = [torch.tensor(1.0), torch.tensor(0.2)]
    runner.invoke(app, ["diagnose", "puz_001"])

def test_list_ids(mocker):
    cli.ds.ids = ["p1", "p2"]
    runner.invoke(app, ["list-ids", "--limit", "1"])
    
    with patch("grid_energy.cli.CroissantDataset", side_effect=Exception("Mock Error")):
        runner.invoke(app, ["list-ids"])