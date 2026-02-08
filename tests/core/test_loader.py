import torch
import pytest
from pathlib import Path
from unittest.mock import patch
from grid_energy.data.loader import CroissantDataset, get_dataloader

class TestCroissantDataset:
    @pytest.fixture
    def dummy_data_path(self, tmp_path: Path) -> Path:
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        file_path = gold_dir / "croissant_dataset.pt"
        
        data = {
            "ids": ["puz_1", "puz_2"],
            "grids": torch.zeros((2, 5, 5)),
            "row_hints": torch.ones((2, 2, 5)),
            "col_hints": torch.ones((2, 2, 5)),
            "metadata": {"version": "1.0"}
        }
        torch.save(data, file_path)
        return gold_dir

    def test_dataset_init_failure(self, tmp_path: Path):
        with patch("grid_energy.data.loader.settings") as mock_settings:
            mock_settings.GOLD_DIR = tmp_path / "non_existent"
            with pytest.raises(FileNotFoundError):
                CroissantDataset()

    def test_dataset_length_and_types(self, dummy_data_path: Path):
        with patch("grid_energy.data.loader.settings") as mock_settings:
            mock_settings.GOLD_DIR = dummy_data_path
            
            dataset = CroissantDataset()
            
            assert len(dataset) == 2
            assert dataset.grids.dtype == torch.float32
            assert dataset.row_hints.dtype == torch.float32

    def test_dataset_getitem_structure(self, dummy_data_path: Path):
        with patch("grid_energy.data.loader.settings") as mock_settings:
            mock_settings.GOLD_DIR = dummy_data_path
            
            dataset = CroissantDataset()
            sample = dataset[0]
            
            expected_keys = {"initial_state", "target_grid", "hints", "id"}
            assert set(sample.keys()) == expected_keys
            
            assert sample["target_grid"].shape == (1, 5, 5)
            assert sample["hints"].shape == (2, 2, 5)
            assert sample["initial_state"].shape == (1, 5, 5)
            assert sample["id"] == "puz_1"

    def test_dataloader_integration(self, dummy_data_path: Path):
        with patch("grid_energy.data.loader.settings") as mock_settings:
            mock_settings.GOLD_DIR = dummy_data_path
            
            loader = get_dataloader(batch_size=1, shuffle=False)
            
            batches = list(loader)
            assert len(batches) == 2
            
            first_batch = batches[0]
            assert first_batch["target_grid"].shape == (1, 1, 5, 5)
            assert first_batch["hints"].shape == (1, 2, 2, 5)