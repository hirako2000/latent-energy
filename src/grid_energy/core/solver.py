import torch

class KineticSolver:
    def __init__(self, energy_fn: torch.nn.Module, step_size: float = 0.02, n_steps: int = 30):
        self.energy_fn = energy_fn
        self.step_size = step_size
        self.n_steps = n_steps

    def resolve(self, initial_grid: torch.Tensor, hints: torch.Tensor):
        self.energy_fn.set_context(hints)
        x = initial_grid.detach().clone().requires_grad_(True)
        
        for i in range(self.n_steps):
            with torch.enable_grad():
                energy = self.energy_fn(x).sum()
            
            grad = torch.autograd.grad(energy, x)[0]
            
            noise = torch.randn_like(x) * (0.01 * (1 - i/self.n_steps))
            x = x - (self.step_size * grad) + noise
            x = torch.clamp(x, -1.0, 1.0).detach().requires_grad_(True)
            
        return x