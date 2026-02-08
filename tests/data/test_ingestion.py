from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from grid_energy.data.ingestion import NONOGRAM_SUBSETS, fetch_bronze_data

class TestIngestion:
    @pytest.fixture(autouse=True)
    def mock_env(self, tmp_path: Path) -> Generator[None, None, None]:
        """Setup mock settings and filesystem for all tests in this class."""
        bronze_dir = tmp_path / "bronze"
        
        # We patch settings globally for the class to prevent any real side effects
        with patch("grid_energy.data.ingestion.settings") as mock_settings:
            mock_settings.BRONZE_DIR = bronze_dir
            mock_settings.HF_TOKEN = "fake_token" # noqa: S105, S101
            mock_settings.HF_DATASET_ID = "org/dataset"
            yield

    @patch("grid_energy.data.ingestion.hf_hub_download")
    @patch("grid_energy.data.ingestion.shutil.copy")
    def test_fetch_bronze_data_logic(
        self,
        mock_copy: MagicMock,
        mock_hf: MagicMock,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path
    ) -> None:
        """
        Verify that hf_hub_download is called with correct params
        and shutil.copy moves the 'cached' file to our bronze structure.
        """
        # 1. Setup Mock Behavior
        # Instead of downloading, return a fake local cache path
        fake_cache_path = str(tmp_path / "dummy_cache" / "dummy.parquet")
        mock_hf.return_value = fake_cache_path

        # 2. Execute Logic
        fetch_bronze_data()

        # 3. Verify hf_hub_download logic
        assert mock_hf.call_count == len(NONOGRAM_SUBSETS)
        
        # Verify the first call details
        first_subset = NONOGRAM_SUBSETS[0]
        mock_hf.assert_any_call(
            repo_id="org/dataset",
            filename=f"{first_subset}/test-00000-of-00001.parquet",
            repo_type="dataset",
            token="fake_token" # noqa: S105, S101, S106
        )

        # 4. Verify Filesystem Logic (shutil)
        # Check if shutil.copy was called for each subset target
        assert mock_copy.call_count == len(NONOGRAM_SUBSETS)
        
        # Capture prints to ensure we aren't seeing actual download logs
        captured = capsys.readouterr()
        for subset in NONOGRAM_SUBSETS:
            assert f"Fetching {subset}" in captured.out
            assert "✓ Stored" in captured.out

    @patch("grid_energy.data.ingestion.hf_hub_download")
    def test_fetch_bronze_data_failure_resilience(
        self,
        mock_hf: MagicMock,
        capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify the loop continues even if one download fails."""
        # Force a failure
        mock_hf.side_effect = Exception("HF API Down")

        fetch_bronze_data()

        captured = capsys.readouterr()
        assert "✗ Failed to download" in captured.out
        # Ensure it tried to call it for every subset regardless of failure
        assert mock_hf.call_count == len(NONOGRAM_SUBSETS)