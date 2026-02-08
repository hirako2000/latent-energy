import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from grid_energy.data.processor import (
    normalize_grid,
    robust_parse_solution,
    process_silver_data,
    generate_atlas
)

class TestProcessor:
    @pytest.fixture
    def mock_bronze_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "initialization": ["{'initialization': [['*', '1'], ['0', '*']], 'hints': {'row_hints': [[1], [1]], 'col_hints': [[1], [1]]}}"],
            "sample_answer": ['{"answer": [[1, 0], [0, 1]]}'],
            "file_name": ["puzzle_001"]
        })

    def test_normalize_grid(self) -> None:
        grid = [["*", "1"], [0, "*"]]
        expected = [["0", "1"], ["0", "0"]]
        assert normalize_grid(grid) == expected
        assert normalize_grid(None) == []
        assert normalize_grid("not a list") == []

    def test_robust_parse_solution_variants(self) -> None:
        json_text = 'Some text {"answer": [[1]]} more text'
        assert robust_parse_solution(json_text) == {"answer": [[1]]}
        ast_text = "Result: {'answer': [[1]]}"
        assert robust_parse_solution(ast_text) == {"answer": [[1]]}
        broken_text = '{"answer": [[1, 1]]'
        assert robust_parse_solution(broken_text) is None
        non_str = 123
        assert robust_parse_solution(non_str) is None # type: ignore

    @patch("grid_energy.data.processor.pd.read_parquet")
    @patch("grid_energy.data.processor.pd.DataFrame.to_parquet")
    @patch("grid_energy.data.processor.plt.savefig")
    @patch("grid_energy.data.processor.settings")
    @patch("grid_energy.data.processor.console.print")
    def test_process_silver_data_full_flow(
        self,
        mock_print: MagicMock,
        mock_settings: MagicMock,
        mock_plt: MagicMock,
        mock_to_parquet: MagicMock,
        mock_read_parquet: MagicMock,
        mock_bronze_df: pd.DataFrame,
        tmp_path: Path
    ) -> None:
        mock_settings.BRONZE_DIR = tmp_path / "bronze"
        mock_settings.SILVER_DIR = tmp_path / "silver"
        mock_settings.ROOT_DIR = tmp_path / "root"
        subset_dir = mock_settings.BRONZE_DIR / "subset_1"
        subset_dir.mkdir(parents=True)
        (subset_dir / "test.parquet").touch()
        mock_read_parquet.return_value = mock_bronze_df
        with patch("grid_energy.data.processor.open", mock_open()):
            process_silver_data()
        assert mock_read_parquet.called
        assert mock_to_parquet.called
        assert mock_plt.called

    @patch("grid_energy.data.processor.plt.savefig")
    @patch("grid_energy.data.processor.settings")
    def test_generate_atlas_execution(
        self,
        mock_settings: MagicMock,
        mock_plt: MagicMock,
        tmp_path: Path
    ) -> None:
        mock_settings.ROOT_DIR = tmp_path
        df = pd.DataFrame({
            "hints": [{"row_hints": [[1], [2]], "col_hints": [[1], [1]]}],
            "id": ["test"]
        })
        generate_atlas(df, "test_subset")
        assert mock_plt.called
        assert "complexity" in df.columns
        assert df.iloc[0]["complexity"] == 3

    @patch("grid_energy.data.processor.settings")
    @patch("grid_energy.data.processor.console.print")
    @patch("grid_energy.data.processor.pd.read_parquet")
    def test_process_silver_data_empty_bronze(
        self,
        mock_read: MagicMock,
        mock_print: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path
    ) -> None:
        mock_settings.BRONZE_DIR = tmp_path / "empty_bronze"
        mock_settings.BRONZE_DIR.mkdir()
        mock_settings.SILVER_DIR = tmp_path / "silver"
        process_silver_data()
        assert not mock_read.called
        all_prints = "".join([str(c) for c in mock_print.call_args_list])
        assert "refined" not in all_prints

    @patch("grid_energy.data.processor.pd.read_parquet")
    @patch("grid_energy.data.processor.settings")
    @patch("grid_energy.data.processor.console.print")
    def test_process_silver_invalid_rows(
        self,
        mock_print: MagicMock,
        mock_settings: MagicMock,
        mock_read_parquet: MagicMock,
        tmp_path: Path
    ) -> None:
        mock_settings.BRONZE_DIR = tmp_path / "bronze"
        subset_dir = mock_settings.BRONZE_DIR / "fail_subset"
        subset_dir.mkdir(parents=True)
        (subset_dir / "test.parquet").touch()
        invalid_df = pd.DataFrame({
            "initialization": ["invalid"],
            "sample_answer": ["invalid"],
            "file_name": ["fail"]
        })
        mock_read_parquet.return_value = invalid_df
        process_silver_data()
        all_prints = "".join([str(c) for c in mock_print.call_args_list])
        assert "FAILED COMPLETELY" in all_prints