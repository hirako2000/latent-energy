import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from grid_energy.visualization.energy_visualizer import (
    MultiCellEnergyVisualizer,
    create_multi_cell_atlas
)

class TestMultiCellEnergyVisualizer:

    @pytest.fixture
    def mock_deps(self, tmp_path):
        path_prefix = "grid_energy.visualization.energy_visualizer"
        with patch(f'{path_prefix}.NonogramCNN'), \
             patch(f'{path_prefix}.NonogramEnergy') as MockEnergy, \
             patch(f'{path_prefix}.torch.load'), \
             patch(f'{path_prefix}.settings') as mock_settings:

            mock_settings.ROOT_DIR = tmp_path
            model_dir = tmp_path / "data" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "ebm_best.pt").touch()

            mock_energy_inst = MockEnergy.return_value
            mock_energy_inst.to.return_value = mock_energy_inst
            mock_energy_inst.return_value = torch.tensor([1.0])

            yield {
                "model_path": model_dir / "ebm_best.pt",
                "output_dir": tmp_path / "output"
            }

    def test_initialization_and_device(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self):
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"], mock_deps["output_dir"])
            assert viz.device is not None
        with pytest.raises(FileNotFoundError):
            MultiCellEnergyVisualizer(model_path=Path("invalid"))

    def test_sample_2d_slice(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self):
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"])
            X, Y, Z = viz.sample_2d_slice(viz.target_grid, 0, 1, steps=2)
            assert Z.shape == (2, 2)

    def test_static_plots(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"):
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"], mock_deps["output_dir"])
            
            viz.create_energy_atlas_3x3()
            viz.create_main_with_satellites()
            viz.create_energy_gradient_field_overview()
            
            with patch("grid_energy.visualization.energy_visualizer.divmod") as mock_div:
                mock_div.side_effect = lambda n, d: (0, 4) if mock_div.call_count == 1 else (0, 0)
                viz.create_cell_neighborhood_map()

    def test_dashboard(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self):
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"], mock_deps["output_dir"])
            
            mock_fig = MagicMock()
            mock_fig.to_html.return_value = "PLOT_HTML_STUB"
            
            real_x = np.linspace(0, 1, 10)
            real_y = np.linspace(0, 1, 10)
            real_z = np.ones((10, 10))
            
            with patch.object(viz, 'sample_2d_slice', return_value=(real_x, real_y, real_z)), \
                patch("grid_energy.visualization.energy_visualizer.go.Figure", return_value=mock_fig), \
                patch("grid_energy.visualization.energy_visualizer.go.Bar", return_value=MagicMock()), \
                patch("builtins.open", mock_open()):
                    viz.create_interactive_comparison_dashboard()


    def test_dashboard_exception_path(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self):
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"])
            with patch("grid_energy.visualization.energy_visualizer.go.Figure", side_effect=ValueError):
                assert viz.create_interactive_comparison_dashboard() is None

    def test_atlas_index_and_integration(self, mock_deps):
        with patch("torch.Tensor.to", lambda self, *args, **kwargs: self), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("builtins.open", mock_open()):
            
            viz = MultiCellEnergyVisualizer(mock_deps["model_path"], mock_deps["output_dir"])
            viz.output_dir.mkdir(parents=True, exist_ok=True)
            
            (viz.output_dir / "energy_atlas_3x3.png").touch()
            (viz.output_dir / "interactive_comparison_dashboard.html").touch()
            
            viz.generate_comprehensive_atlas()
            
    def test_create_multi_cell_atlas_wrapper(self, mock_deps):
        path = "grid_energy.visualization.energy_visualizer.MultiCellEnergyVisualizer"
        with patch(path) as MockViz:
            MockViz.return_value.generate_comprehensive_atlas.return_value = {}
            assert create_multi_cell_atlas() is True
            
            MockViz.side_effect = Exception
            assert create_multi_cell_atlas() is False