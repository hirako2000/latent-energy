import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Any

from grid_energy.data.synthesizer import (
    to_list,
    analyze_dataset_dimensions,
    process_puzzle_row,
    save_gold_dataset,
    synthesize_gold,
    verify_croissant
)

@pytest.fixture
def sample_silver_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "size": 5,
            "solution": [["s", "·", "s", "·", "·"]] + [["·"]*5]*4,
            "hints": {
                "row_hints": [[1, 1], [0], [0], [0], [0]],
                "col_hints": [[1], [0], [1], [0], [0]]
            }
        },
        {
            "size": 10,
            "solution": [["s"]*10]*10,
            "hints": {"row_hints": [[10]], "col_hints": [[10]]}
        }
    ])

# --- Logic Tests ---

def test_to_list_variations():
    assert to_list(None) == []
    assert to_list(np.array([1, 2])) == [1, 2]
    assert to_list([10, 20]) == [10, 20]

def test_analyze_dataset_dimensions(sample_silver_df: pd.DataFrame):
    included, max_grid, max_hint = analyze_dataset_dimensions(sample_silver_df)
    assert included is not None
    assert len(included) == 1
    assert max_grid == 5
    assert max_hint == 2

def test_analyze_dataset_dimensions_no_match():
    df = pd.DataFrame([{"size": 10, "hints": {}}])
    included, max_grid, max_hint = analyze_dataset_dimensions(df)
    assert included is None
    assert max_grid == 0

def test_process_puzzle_row_mapping():
    num_samples, size, hint_len = 1, 5, 2
    grids = np.zeros((num_samples, size, size), dtype=np.int8)
    rows = np.zeros((num_samples, size, hint_len), dtype=np.int16)
    cols = np.zeros((num_samples, size, hint_len), dtype=np.int16)
    
    row_data: dict[str, Any] = {
        "size": 5,
        "solution": [["s", "·", "·", "·", "·"]] * 5,
        "hints": {
            "row_hints": [[1], [1], [1], [1], [1]],
            "col_hints": [[5], [], [], [], []]
        }
    }
    process_puzzle_row(0, row_data, grids, rows, cols)
    assert np.all(grids[0, :, 0] == 1)
    assert rows[0, 0, 0] == 1
    assert cols[0, 0, 0] == 5

def test_save_gold_dataset_io(mocker: Any, tmp_path: Path):
    mock_settings = mocker.patch("grid_energy.data.synthesizer.settings")
    mock_settings.GOLD_DIR = tmp_path
    mock_save = mocker.patch("grid_energy.data.synthesizer.torch.save")
    
    grids = np.zeros((1, 5, 5))
    r_h = np.zeros((1, 5, 1))
    c_h = np.zeros((1, 5, 1))
    
    data = save_gold_dataset(grids, r_h, c_h, 1, 5, 1, {"5x5": 1})
    
    assert data["metadata"]["samples"] == 1
    mock_save.assert_called_once()
    assert str(tmp_path) in str(mock_save.call_args[0][1])

def test_synthesize_gold_full_pipeline(mocker: Any, tmp_path: Path, sample_silver_df: pd.DataFrame):
    mock_settings = mocker.patch("grid_energy.data.synthesizer.settings")
    mock_settings.SILVER_DIR = tmp_path / "silver"
    mock_settings.GOLD_DIR = tmp_path / "gold"
    mock_settings.SILVER_DIR.mkdir()
    
    (mock_settings.SILVER_DIR / "test.parquet").touch()
    
    mocker.patch("grid_energy.data.synthesizer.pd.read_parquet", return_value=sample_silver_df)
    mocker.patch("grid_energy.data.synthesizer.torch.save")
    mock_viz = mocker.patch("grid_energy.data.synthesizer.run_visualizer")
    
    synthesize_gold()
    
    assert mock_viz.called

def test_synthesize_gold_empty_silver(mocker: Any, tmp_path: Path):
    mock_settings = mocker.patch("grid_energy.data.synthesizer.settings")
    mock_settings.SILVER_DIR = tmp_path / "silver_empty"
    mock_settings.GOLD_DIR = tmp_path / "gold"
    mock_settings.SILVER_DIR.mkdir() # Exists but empty
    
    mock_print = mocker.patch("grid_energy.data.synthesizer.console.print")
    
    synthesize_gold()
    
    assert any("No silver files found" in str(call) for call in mock_print.call_args_list)

def test_verify_croissant_from_disk(mocker: Any, tmp_path: Path):
    mock_settings = mocker.patch("grid_energy.data.synthesizer.settings")
    mock_settings.GOLD_DIR = tmp_path
    dataset_path = tmp_path / "croissant_dataset.pt"
    dataset_path.touch() # Make path.exists() True
    
    mock_load = mocker.patch("grid_energy.data.synthesizer.torch.load", return_value={
        "grids": torch.ones((1, 5, 5)),
        "row_hints": torch.ones((1, 5, 2)),
        "ids": ["p_0000"],
        "metadata": {"max_grid_size": 5}
    })
    
    mocker.patch("grid_energy.data.synthesizer.console.print")
    verify_croissant(data=None)
    
    assert mock_load.called

def test_verify_croissant_data_passed(mocker: Any):
    mock_data = {
        "grids": torch.ones((1, 5, 5)),
        "row_hints": torch.ones((1, 5, 2)),
        "ids": ["p_0000"],
        "metadata": {"max_grid_size": 5}
    }
    mock_print = mocker.patch("grid_energy.data.synthesizer.console.print")
    verify_croissant(data=mock_data)
    
    # Check that we actually tried to print the grid strings
    assert mock_print.call_count > 1