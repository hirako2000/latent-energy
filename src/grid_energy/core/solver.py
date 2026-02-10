import torch

DEFAULT_STEP_SIZE = 0.03
DEFAULT_N_STEPS = 300
DEFAULT_DETERMINISTIC = True
NOISE_INITIAL_SCALE = 0.1
NOISE_DECAY_POWER = 2
CLAMP_MIN = -2.0
CLAMP_MAX = 2.0
GRADIENT_ENABLED = True
GRADIENT_RETURN_FIRST_ONLY = True


class KineticSolver:
    def __init__(
        self,
        energy_fn: torch.nn.Module,
        step_size: float = DEFAULT_STEP_SIZE,
        n_steps: int = DEFAULT_N_STEPS,
        deterministic: bool = DEFAULT_DETERMINISTIC
    ):
        self.energy_fn = energy_fn
        self.step_size = step_size
        self.n_steps = n_steps
        self.deterministic = deterministic

    def calculate_noise_scale(self, step_index: int) -> float:
        progress_ratio = step_index / self.n_steps
        decay_factor = (1 - progress_ratio) ** NOISE_DECAY_POWER
        return NOISE_INITIAL_SCALE * decay_factor

    def resolve(self, initial_grid: torch.Tensor, hints: torch.Tensor):
        self.energy_fn.set_context(hints) # type: ignore
        x = initial_grid.detach().clone().requires_grad_(True)

        for i in range(self.n_steps):
            with torch.enable_grad():
                energy = self.energy_fn(x).sum()

            grad = torch.autograd.grad(energy, x, create_graph=GRADIENT_ENABLED)[0]

            if not self.deterministic:
                noise_scale = self.calculate_noise_scale(i)
                noise = torch.randn_like(x) * noise_scale
                x = x - (self.step_size * grad) + noise
            else:
                x = x - (self.step_size * grad)

            x = torch.clamp(x, CLAMP_MIN, CLAMP_MAX).detach().requires_grad_(True)

        return x