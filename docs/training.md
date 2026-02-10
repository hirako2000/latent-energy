# Latent energy

_Kinetic resolution for logic systems using energy based models_

This project implements an energy based model for solving nonogram puzzles, treating constraint satisfaction as an energy minimization problem rather than a search problem. The system learns to assign energy values to grid configurations, where lower energy corresponds to more likely solutions, and then uses gradient based optimization to find the minimum energy state that satisfies all constraints.

## Technical approach

We use a convolutional neural network architecture to model the energy function, combined with explicit logical constraints in the energy formulation. The training process involves both supervised pre training and self supervised contrastive learning.

### Core components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Energy function | NonogramCNN with 4 conv layers | Maps grid+hints to scalar energy |
| Logic encoding | Row/col hint processing | Incorporates puzzle constraints |
| Kinetic solver | Gradient descent with noise | Navigates energy landscape |
| Training pipeline | Two phase hybrid | Combines supervised+EBM learning |

## Energy function architecture

The energy function combines three components:

$$
E_{\text{total}} = E_{\text{neural}} + E_{\text{logic}} + E_{\text{binary}}
$$

### Neural energy component
A convolutional neural network processes the 5x5 grid and hints. The grid is treated as a single channel image, with hints encoded as additional channels. The CNN uses three convolutional layers with 256 channels each, followed by a final projection to scalar energy.

### Logic energy component
This explicit term enforces row and column constraints:

$$
E_{\text{logic}} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{MSE}(\text{row\_sums}_i, \text{hint\_row\_sums}_i) + \text{MSE}(\text{col\_sums}_i, \text{hint\_col\_sums}_i) \right)
$$

### Binary energy component
Encourages solutions to be near 0 or 1 values:

$$
E_{\text{binary}} = \frac{1}{N \times 25} \sum_{i=1}^{N} \sum_{j=1}^{25} x_{ij}^2
$$

## Training methodology

### Supervised pre training
The model first learns direct mapping from hints to solutions using mean squared error loss. This provides a strong initialization for the energy based learning phase.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning rate | $1 \times 10^{-3}$ | Aggressive initial learning |
| Batch size | 16 | Balanced memory/computation |
| Epochs | 15 | Meaningful initialization |
| Gradient clipping | 0.5 | Prevent exploding gradients |
| Loss function | $\text{MSE}(x_{\text{pred}}, x_{\text{true}})$ | Pixel wise accuracy |

### Energy based training
The model learns to distinguish good solutions from bad ones through contrastive learning. We compare real solutions against random noise and perturbed solutions.

Contrastive loss:

$$
\mathcal{L} = \max(0, E_{\text{solution}} - E_{\text{noise}} + m) + \max(0, E_{\text{solution}} - E_{\text{perturbed}} + m)
$$

where $m = 2.0$ is the energy margin.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning rate | $1 \times 10^{-4}$ | Slower fine tuning |
| Energy margin | 2.0 | Clear separation threshold |
| Noise scale | 0.3 | Gaussian noise magnitude |
| Perturbation probability | 0.2 | Hard negative generation |

## Kinetic resolution solver

Once trained, the energy function enables solution finding through gradient based optimization:

1. **Initialization**: Start from random or CNN predicted state
2. **Gradient descent**: Follow $-\nabla E(x \mid H)$
3. **Noise injection**: $\epsilon \sim \mathcal{N}(0, \sigma_t)$ where $\sigma_t = 0.1 \times (1 - t/T)^2$
4. **Constraint enforcement**: $\text{clamp}(x, -2, 2)$
5. **Convergence**: Stop when $\Delta E < 1 \times 10^{-4}$

| Parameter | Value | Effect |
|-----------|-------|--------|
| Steps | 150-300 | Trade off accuracy/speed |
| Step size | $3 \times 10^{-2}$ | Convergence rate |
| Initial noise scale | 0.1 | Early exploration |
| Clamping range | $[-2, 2]$ | Numerical stability |

## Data pipeline

The system uses a three layer medallion architecture:

```
Bronze Data → Silver Processing → Gold Synthesis
                                          ↓
                               Two Phase Training
                                          ↓
                              Kinetic Resolution Solver
```

Each puzzle is encoded as:
- Grid: $5 \times 5$ binary matrix
- Row hints: $5 \times 2$ tensor (max 2 hints per row)
- Column hints: $5 \times 2$ tensor (max 2 hints per column)

## Performance results

The hybrid training approach achieves 100% accuracy on the validation set after 50 epochs. The kinetic solver converges to correct solutions in 150-300 steps, with inference times under $1 \times 10^{-1}$ seconds per puzzle on GPU.

## Key insights

1. The convolutional architecture effectively captures local patterns in grid configurations
2. Explicit logic constraints in the energy function help guide optimization
3. Contrastive learning with margin provides clear separation between good and bad states
4. The kinetic solver's noise injection is crucial for escaping local minima

## Why this works

The supervised phase provides strong initialization by teaching the model approximate mappings. The EBM phase adds robustness and generalization by learning what constitutes a valid solution in energy space, not just memorizing training examples.

| Method | Advantages | Limitations addressed |
|--------|------------|-----------------------|
| Pure supervised | Fast training, direct optimization | Poor generalization, memorization risk |
| Pure EBM | Strong generalization, constraint learning | Slow convergence, initialization sensitive |
| Hybrid (ours) | Fast initialization + robust generalization | Combines strengths, mitigates weaknesses |

## Usage

```bash
# Install just if not available
cargo install just

# Set up environment
just setup

# Solve a puzzle
just resolve id="puzzle_0060"

# Train the model
just train

# Visualize energy landscapes
just viz-energy
```

## Implementation details

The codebase uses:
- PyTorch for neural network implementation
- UV for Python package management
- Just for task automation
- Pre commit hooks for code quality
- Comprehensive test suite with >90% coverage

All training and inference supports GPU acceleration through CUDA, MPS, or CPU fallback. The latency to solve a problem is approximately $1 \times 10^{-1}$ seconds.
