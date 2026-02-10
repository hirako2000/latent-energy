import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from grid_energy.data.visualization.gold_visualizer import (
    to_list,
    generate_gold_occupancy_heatmap,
    generate_gold_summary_metrics_png,
    run_visualizer
)

@pytest.fixture
def mock_metrics() -> dict[str, Any]:
    return {
        "size_metrics": {
            "5x5": {
                "count": 10,
                "avg_fill_rate": 0.45,
                "avg_complexity": 15.5,
                "avg_hint_count": 8.2,
                "max_hint_value": 5.0
            }
        },
        "tensor_info": {
            "shape": [10, 5, 5],
            "global_fill_rate": 0.45
        }
    }

def test_to_list():
    assert to_list(None) == []
    assert to_list(np.array([1, 2])) == [1, 2]
    assert to_list([3, 4]) == [3, 4]

@patch("grid_energy.data.visualization.gold_visualizer.plt")
def test_generate_gold_occupancy_heatmap(mock_plt: MagicMock, tmp_path: Path):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    grids = np.zeros((5, 5, 5))
    generate_gold_occupancy_heatmap(grids, tmp_path)
    
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()

@patch("grid_energy.data.visualization.gold_visualizer.plt")
def test_generate_gold_summary_metrics_png(mock_plt: MagicMock, mock_metrics: dict, tmp_path: Path):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_table = MagicMock()
    mock_ax.table.return_value = mock_table
    mock_table.get_celld.return_value = {(0, 0): MagicMock(), (1, 0): MagicMock()}
    
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    generate_gold_summary_metrics_png(mock_metrics, tmp_path)
    
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()

def test_run_visualizer_no_files(mocker: Any, tmp_path: Path):
    mock_settings = mocker.patch("grid_energy.data.visualization.gold_visualizer.settings")
    mock_settings.SILVER_DIR = tmp_path / "silver"
    mock_settings.SILVER_DIR.mkdir()
    
    mock_print = mocker.patch("grid_energy.data.visualization.gold_visualizer.console.print")
    run_visualizer()
    
    assert any("No silver files found" in str(call) for call in mock_print.call_args_list)

def test_run_visualizer_success(mocker: Any, tmp_path: Path):
    mock_settings = mocker.patch("grid_energy.data.visualization.gold_visualizer.settings")
    mock_settings.SILVER_DIR = tmp_path / "silver"
    mock_settings.ROOT_DIR = tmp_path / "root"
    mock_settings.SILVER_DIR.mkdir(parents=True)
    (mock_settings.SILVER_DIR / "data.parquet").touch()
    
    valid_df = pd.DataFrame([{
        "size": 5,
        "solution": [["s", "·", "·", "·", "·"]] * 5,
        "hints": {"row_hints": [[1]]*5, "col_hints": [[5]] + [[]]*4}
    }])
    
    # mocks of anything that touches the screen or disk
    mocker.patch("grid_energy.data.visualization.gold_visualizer.pd.read_parquet", return_value=valid_df)
    mocker.patch("grid_energy.data.visualization.gold_visualizer.plt.subplots", return_value=(MagicMock(), MagicMock()))
    mocker.patch("grid_energy.data.visualization.gold_visualizer.plt.savefig")
    mocker.patch("grid_energy.data.visualization.gold_visualizer.plt.close")
    mocker.patch("grid_energy.data.visualization.gold_visualizer.sns.heatmap")
    mocker.patch("grid_energy.data.visualization.gold_visualizer.console.print")
    
    run_visualizer()
    
    viz_dir = mock_settings.ROOT_DIR / "docs" / "visualizations"
    assert (viz_dir / "gold_metrics.json").exists()