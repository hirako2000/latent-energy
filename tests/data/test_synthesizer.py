import torch
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from grid_energy.data.synthesizer import synthesize_gold, to_list, verify_croissant

class TestSynthesizer:
    @pytest.fixture
    def mock_silver_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "size": [5, 5],
            "solution": [
                [['s', '.', '.', '.', '.'], ['.', 's', '.', '.', '.']],
                [['.', '.', 's', '.', '.'], ['.', '.', '.', 's', '.']]
            ],
            "hints": [
                {"row_hints": [[1], [1]], "col_hints": [[1], [1]]},
                {"row_hints": [[1], [1]], "col_hints": [[1], [1]]}
            ]
        })

    def test_to_list_conversion(self) -> None:
        assert to_list(None) == []
        assert to_list(np.array([1, 2])) == [1, 2]
        assert to_list([3, 4]) == [3, 4]

    @patch("grid_energy.data.synthesizer.pd.read_parquet")
    @patch("grid_energy.data.synthesizer.torch.save")
    @patch("grid_energy.data.synthesizer.plt.savefig")
    @patch("grid_energy.data.synthesizer.settings")
    @patch("grid_energy.data.synthesizer.verify_croissant")
    def test_synthesize_gold_execution(
        self,
        mock_verify: MagicMock,
        mock_settings: MagicMock,
        mock_plt: MagicMock,
        mock_torch_save: MagicMock,
        mock_read_parquet: MagicMock,
        mock_silver_df: pd.DataFrame,
        tmp_path: Path
    ) -> None:
        mock_settings.SILVER_DIR = tmp_path / "silver"
        mock_settings.GOLD_DIR = tmp_path / "gold"
        mock_settings.ROOT_DIR = tmp_path / "root"
        mock_settings.SILVER_DIR.mkdir()
        (mock_settings.SILVER_DIR / "test.parquet").touch()
        mock_read_parquet.return_value = mock_silver_df
        with patch("builtins.open", mock_open()):
            synthesize_gold()
        assert mock_torch_save.called
        assert mock_plt.called
        assert mock_verify.called
        saved_data = mock_torch_save.call_args[0][0]
        assert "grids" in saved_data
        assert "metadata" in saved_data
        assert saved_data["metadata"]["samples"] == 2

    @patch("grid_energy.data.synthesizer.settings")
    @patch("grid_energy.data.synthesizer.console.print")
    def test_synthesize_no_data_handling(
        self,
        mock_print: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path
    ) -> None:
        mock_settings.SILVER_DIR = tmp_path / "empty"
        mock_settings.SILVER_DIR.mkdir()
        synthesize_gold()
        called_arg = str(mock_print.call_args_list[0][0][0])
        assert "No silver files found" in called_arg

    @patch("grid_energy.data.synthesizer.console.print")
    def test_verify_croissant_output(self, mock_print: MagicMock) -> None:
        mock_data = {
            "grids": torch.ones((1, 5, 5)),
            "row_hints": torch.ones((1, 5, 5)),
            "col_hints": torch.ones((1, 5, 5)),
            "ids": ["test_puzzle"],
            "metadata": {"max_grid_size": 5}
        }
        verify_croissant(data=mock_data)
        
        found_panel = False
        found_size = False
        
        for call in mock_print.call_args_list:
            arg = call[0][0]
            if hasattr(arg, "renderable"):
                content = str(arg.renderable)
            else:
                content = str(arg)
                
            if "Verification Sample ID" in content:
                found_panel = True
            if "Puzzle size: 5x5" in content:
                found_size = True
                
        assert found_panel
        assert found_size