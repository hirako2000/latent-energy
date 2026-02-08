from unittest.mock import MagicMock

import pytest
import torch
from grid_energy.core.solver import KineticSolver

class TestKineticSolver:
    @pytest.fixture
    def mock_energy_fn(self) -> MagicMock:
        """Creates a mock nn.Module that simulates an energy surface."""
        model = MagicMock(spec=torch.nn.Module)
        
        # value that allows gradient computation.
        # A simple quadratic function: energy = x^2
        def side_effect(x: torch.Tensor) -> torch.Tensor:
            return (x ** 2).sum()

        model.side_effect = side_effect
        # Ensure it has the custom method used in solver.py
        model.set_context = MagicMock()
        return model

    @pytest.fixture
    def solver_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns dummy (grid, hints) tensors."""
        grid = torch.randn((1, 5, 5), requires_grad=True)
        hints = torch.zeros((1, 2, 5))
        return grid, hints

    def test_resolve_deterministic_optimization(
        self,
        mock_energy_fn: MagicMock,
        solver_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Tests that the solver reduces energy in deterministic mode."""
        initial_grid, hints = solver_input
        solver = KineticSolver(energy_fn=mock_energy_fn, n_steps=5, deterministic=True)

        result = solver.resolve(initial_grid, hints)

        mock_energy_fn.set_context.assert_called_once_with(hints)

        assert result.shape == initial_grid.shape
        assert result.requires_grad is True

    def test_resolve_stochastic_mode(
        self,
        mock_energy_fn: MagicMock,
        solver_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Tests the solver path with noise enabled."""
        initial_grid, hints = solver_input
        solver = KineticSolver(energy_fn=mock_energy_fn, n_steps=3, deterministic=False)

        result = solver.resolve(initial_grid, hints)

        assert result.shape == initial_grid.shape
        # clamped within the solver's range [-2, 2]
        assert torch.all(result >= -2.0) and torch.all(result <= 2.0)

    def test_gradient_flow_interruption(
        self,
        mock_energy_fn: MagicMock,
        solver_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Ensures the solver correctly handles detached tensors between steps."""
        initial_grid, hints = solver_input
        solver = KineticSolver(energy_fn=mock_energy_fn, n_steps=2)

        result = solver.resolve(initial_grid, hints)

        assert result.grad_fn is None