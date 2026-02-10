# Latent Energy


<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14+-blue.svg?style=for-the-badge" alt="Python 3.14+">
  </a>
  <a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json&style=for-the-badge" alt="Ruff">
  </a>
</div>

### A model to solve nonograms

<div align="center">
  <img src="./img/output.gif" alt="Grid Energy Solver Demo" width="380">
</div>

#### Kinetic resolution for logic systems using energy based models

This project implements an [energy based model](https://en.wikipedia.org/wiki/Energy-based_model) for solving nonogram puzzles, treating constraint satisfaction as an energy minimization problem rather than a search problem. The system learns to assign energy values to grid configurations, where lower energy corresponds to more likely solutions, and then uses gradient based optimization to find the minimum energy state that satisfies all constraints.

# Table of Contents

- [The technical approach](#technical-approach)
- [Energy function](#energy-function)
- [Data pipeline](#data-pipeline)
- [Training](#training)
- [Kinetic resolution solver](#kinetic-resolution-solver)
- [Performance](#performance)
- [Key insights](#key-insights)
- [Usage](#usage)
- [Library and Hardware](#library-and-hardware)

## Technical approach

We implement a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to model the energy function, combined with explicit logical constraints in the energy formulation. The training process involves both supervised pre training and self supervised contrastive learning.

### Core components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Energy function | NonogramCNN with 4 Conv2d layers| Maps grid+hints to scalar energy |
| Logic encoding | Row/col hint processing | Incorporates puzzle constraints |
| Kinetic solver | Gradient descent with noise | Navigates energy landscape |
| Training pipeline | Two phase hybrid | Combines supervised+EBM learning |

## Energy function

The energy function combines three components:

```
E_total = E_neural + E_logic + E_binary
```

### Neural energy

A convolutional neural network processes the 5x5 grid and hints. The grid is treated as a single channel, with hints encoded as additional channels. The CNN uses convolutional layers with 256 channels each, followed by a final projection to scalar energy.

### Logic energy

This enforces row and column constraints:

```
E_logic = MSE(row_sums, hint_row_sums) + MSE(col_sums, hint_col_sums)
```

### Binary energy

Encourages solutions to be near 0 or 1 values:

```
E_binary = mean(grid²)
```

## Data pipeline

The system uses a three layer [medallion architecture](https://www.databricks.com/glossary/medallion-architecture):

**Bronze**: Raw parquet files from Hugging Face, see [ingestion.py](./src/grid_energy/data/ingestion.py)
**Silver**: Processed puzzles with normalized schemas, in [processor.py](./src/grid_energy/data/processor.py)
**Gold**: Tensor representations for training, via [synthesizer.py](./src/grid_energy/data/synthesizer.py)


Each puzzle is encoded as:

- Grid: 5x5 binary matrix
- Row hints: 5x2 tensor (max 2 hints per row)
- Column hints: 5x2 tensor (max 2 hints per column)

### Data Visualizations

For the processed data, we generate visualizations for at-a-glance verification

<table>
    <tr>
      <td> <p align="center"><b>Summary Metrics</b></p> <img src="docs/visualizations/gold_summary_metrics.png" width="100%"> </td> 
      <td> <p align="center"><b>Silver Complexity vs Size</b></p> <img src="docs/visualizations/silver_complexity_vs_size.png" width="100%"> </td> 
    </tr>
    <tr> 
      <td> <p align="center"><b>Silver Size Distribution</b></p> <img src="docs/visualizations/silver_size_distribution.png" width="100%"> </td>
      <td> <p align="center"><b>5x5 Complexity Atlas</b></p> <img src="docs/visualizations/nonogram_5x5_complexity_atlas.png" width="100%"> </td>
    </tr>
    <tr>
      <td width="50%"> <p align="center"><b>Complexity Density</b></p> <img src="docs/visualizations/gold_complexity_density.png" width="100%"> </td>
      <td width="50%"> <p align="center"><b>Hint Count Distribution</b></p> <img src="docs/visualizations/gold_hint_count_distribution.png" width="100%"> </td> 
    </tr>
    <tr>
      <td> <p align="center"><b>Hint Value Frequency</b></p> <img src="docs/visualizations/gold_hint_value_frequency.png" width="100%"> </td>
      <td> <p align="center"><b>Occupancy Heatmap</b></p> <img src="docs/visualizations/gold_occupancy_heatmap.png" width="100%"> </td>
    </tr> 

 </table>

Read more about [data-engineering](./docs/data-engineering.md)

## Training

A two phases approach.

### Supervised pre training

The model first learns direct mapping from hints to solutions using mean squared error loss. This provides a strong head start for the energy based learning phase.

Parameters:
- Learning rate: 1e-3
- Batch size: 16
- Epochs: 15
- Loss: MSE between predicted and target grids

### Energy based training

The model learns to distinguish good solutions from bad ones through contrastive learning. We compare real solutions against random noise and perturbed solutions.

Contrastive loss:
```
loss = max(0, E_solution - E_noise + margin) + 
       max(0, E_solution - E_perturbed + margin)
```

Parameters:

- Learning rate: 1e-4
- Energy margin: 2.0
- Noise scale: 0.3
- Perturbation probability: 0.2


### Model Visualizations

<table>
  <tr>
    <td width="50%">
      <p align="center"><b>Gradient Field Overview</b></p>
      <img src="docs/energy_atlas/gradient_field_overview.png" width="100%">
    </td>
    <td width="50%">
      <p align="center"><b>Cell Neighborhood Map</b></p>
      <img src="docs/energy_atlas/cell_neighborhood_map.png" width="100%">
    </td>
  </tr>

  <tr>
    <td width="50%">
      <p align="center"><b>Energy Atlas (3x3 Detail)</b></p>
      <img src="docs/energy_atlas/energy_atlas_3x3.png" width="100%">
    </td>
    <td width="50%">
      <p align="center"><b>Cell Occupancy Heatmap</b></p>
      <img src="img/gold_cell_occupancy.png" width="100%">
    </td>
  </tr>
</table>

Read more about [training](./docs/training.md)

## Kinetic resolution solver

Once trained, the energy function enables solution finding through gradient based optimization:

1. **Initialization**: Start from random or CNN predicted state
2. **Gradient descent**: Follow negative energy gradient
3. **Noise injection**: Stochastic term helps escape local minima
4. **Constraint enforcement**: Clamp values to valid range
5. **Convergence**: Stop when energy stabilizes

Solver parameters:
- Steps: 150 300
- Step size: 0.03
- Noise scale: 0.1 with decay
- Clamping range: [-2, 2]

You can see the cliff and valleys of the gradient descent in 3D, see those [interactive](https://latent-energy.surge.sh/interactive_comparison_dashboard.html) graphs

Or the dashboard below
<br/>
<a href="https://latent-energy.surge.sh">
  <img src="./img/3d-viz.png" width="100%" alt="3D Visualization">
</a>


## Performance

The hybrid training approach achieves 100 percent accuracy on the validation set after 50 epochs. The kinetic solver converges to correct solutions in 150 300 steps, with inference times under 100 milliseconds per puzzle on GPU.

1. The convolutional architecture effectively captures local patterns in the grid configurations
2. Explicit logic constraints in the energy function help guide the optimization
3. Contrastive learning with margin provides clear separation between good and bad states
4. The kinetic solver's noise injection is crucial for escaping local minima

## Usage

For convenience, install [just](https://github.com/casey/just)

### Prerequisites

- [Python](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) installed

```bash
# install dependencies
just setup

# Fetch and prepare data
just inject bronze
just prepare-gold

# Train model
just train

# Solve a puzzle
just resolve puzzle_0011 # any puzzle from 0000 to 0099

# Build 3d interactive energy landscapes
just viz-energy
```

Alternatively you may use `uv` commands directly

```bash
# prepare data
uv run grid-energy ingest bronze
uv run grid-energy ingest silver
uv run grid-energy ingest gold

# train model
uv run grid-energy train --epochs 50

# Solve any puzzle of dataset
uv run grid-energy resolve --puzzle-id="puzzle_0013"
```

#### Dev utilities 

```bash
# fix linting issues
just fix

# Check pyright
just check # this will also run the tests

# Run the tests
just test
```


## Library and Hardware

The codebase uses: [PyTorch](https://pytorch.org/) for neural network implementation. [UV](https://docs.astral.sh/uv/) for Python package management.
[Ruff](https://docs.astral.sh/ruff/) and [Pyright](https://github.com/microsoft/pyright) for linting and types.
[Just](https://github.com/casey/just) for short-hand UV commands.

Unit test suites targetting 90+ percent coverage, uses [Pytest](https://docs.pytest.org/en/stable/).

All training and inference supports GPU acceleration through CUDA, MPS, or CPU fallback. The project was tested on Apple Silicon with both GPU and CPU fallback.
