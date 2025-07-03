# Thalamocortical Network Optimization for Fast Initial State Preparation

This repository implements optimization methods for designing thalamocortical neural networks that can rapidly decay to baseline activity. The key innovation is using low-rank perturbations to modify network connectivity matrices to achieve desired dynamical properties.

## Table of Contents
1. [Mathematical Background](#mathematical-background)
2. [Method Overview](#method-overview)
3. [Code Structure](#code-structure)
4. [Learning Path](#learning-path)
5. [Implementation Details](#implementation-details)
6. [Running the Code](#running-the-code)

## Mathematical Background

### 1. Network Dynamics

The core model is a linear dynamical system representing neural activity:

```
τ ẋ = -x + Jx
```

Where:
- `x(t)` is the N-dimensional neural activity vector
- `J` is the N×N connectivity matrix
- `τ` is the neural time constant

The solution to this system is:
```
x(t) = exp((J-I)t/τ) x₀ = R exp(Γt) L x₀
```

Where:
- `R` and `L` are right and left eigenvector matrices of `J`
- `Γ = (Λ-I)/τ` with `Λ` being the diagonal eigenvalue matrix
- `x₀` is the initial condition

### 2. Low-Rank Perturbation Structure

The key insight is to decompose the connectivity matrix as:
```
J = J₀ + UV
```

Where:
- `J₀` is a random N×N matrix with elements drawn from N(0, 1/N)
- `U` is an N×n matrix
- `V` is an n×N matrix
- `n << N` (typically n/N ∈ [0.03, 0.5])

This low-rank structure allows efficient control of network dynamics while keeping the number of parameters manageable.

### 3. Optimization Objective

The goal is to find `U` and `V` that minimize a composite loss function:

#### Loss Component 1: Decay Speed (L₁)
Minimizes the integrated squared norm of activity:
```
L₁(U,V) = (1/Nτ) 𝔼[∫₀^∞ ||x(t)||² dt]
        = (1/Nτ) Tr(R(LL^H ∘ F)R^H)
```

Where `F_{ij} = -1/(γᵢ + γⱼ*)` with `γᵢ` being eigenvalues of `(J-I)/τ`.

#### Loss Component 2: Stability Constraint (L₂)
Ensures all eigenvalues have real parts less than 1:
```
L₂(U,V) = Σᵢ max(Re(γᵢ), 0)²
```

#### Loss Component 3: Smoothness Constraint (L₃)
Encourages smooth output dynamics for `y = w^T x`:
```
L₃(U,V) = τ w^T R(LL^H ∘ G)R^H w
```

Where `G_{ij} = γᵢγⱼ* F_{ij}`.

The total loss is:
```
ℒ(U,V) = L₁(U,V) + α L₂(U,V) + β L₃(U,V)
```

### 4. Custom Eigendecomposition Gradient

Since TensorFlow's built-in eigendecomposition is slow, the code implements a custom gradient using:

For matrix `A` with eigenvalues `Λ` and eigenvectors `V`:
```
∇_{A*} = V^{-H}[∇_{Λ*} + F* ∘ (V^H ∇_{V*} - V^H V(V^H ∇_{V*} ∘ I))]V^H
```

Where `F_{ii} = 0` and `F_{ij} = (λⱼ - λᵢ)⁻¹` for `i ≠ j`.

## Method Overview

### Optimization Strategy

1. **Initialization**: Start with random small-magnitude `U` and `V`
2. **Two-stage optimization**:
   - **Stage 1**: ADAM optimizer for initial rough optimization (30 epochs)
   - **Stage 2**: L-BFGS for fine-tuning (up to 200 iterations)

This two-stage approach is necessary because L-BFGS can be unstable when starting from random initialization.

### Key Parameters

- **Network size (N)**: Typically 100-500 neurons
- **Rank fraction (n/N)**: 0.03-0.5 (higher rank = more control but more parameters)
- **Time constant (τ)**: Neural dynamics timescale
- **Smoothness weight (β)**: Controls output smoothness vs decay speed tradeoff
- **Stability weight (α)**: Large value (10⁵) to strongly enforce stability

## Code Structure

```
thalamocortical/
├── my_lbfgs.py                    # Core optimization routines
├── myutils.py                     # Utility functions
├── train_models.py                # Command-line training script
├── Preparatory network optimization.ipynb  # Main demo notebook
├── Preparatory network optimization - statistics.ipynb  # Statistical analysis
├── Stability of eigenvalue control.ipynb  # Stability analysis
└── prep_opt.ipynb                 # Interactive optimization
```

### Key Files Explained

- **`my_lbfgs.py`**: Contains the mathematical core:
  - `myeig()`: Custom eigendecomposition with gradient
  - `loss_fx()`: Loss function implementation
  - `do_adam()`, `do_lbfgs()`: Optimization routines

- **`train_models.py`**: Batch training script for parameter sweeps

- **Notebooks**: Interactive demonstrations and analyses

## Learning Path

### For Theoreticians

1. **Start with the math**:
   - Read through the Mathematical Background section above
   - Study `Preparatory network optimization.ipynb` for derivations
   - Focus on understanding the loss function components

2. **Key theoretical insights**:
   - Low-rank perturbations can dramatically change dynamics
   - Eigenvalue placement determines decay rates
   - Trade-off between decay speed and output smoothness

3. **Stability analysis**:
   - Review `Stability of eigenvalue control.ipynb`
   - Understand condition number dependencies

### For Practitioners

1. **Quick start**:
   ```python
   # Simple example
   from my_lbfgs import build_model, do_adam, do_lbfgs
   
   model = build_model(N=100, n_fracs=[0.1], K=1, tau=0.01, alpha=1e5, betas=[0.01])
   param, losses = do_adam(model, epochs=30, learning_rate=0.01)
   param, losses, results = do_lbfgs(model, iters=200, param=param)
   ```

2. **Parameter tuning**:
   - Start with N=100 for quick experiments
   - Try n_fracs in [0.05, 0.1, 0.2]
   - Adjust β for smoothness vs speed trade-off

3. **Visualization**:
   - Use notebooks for interactive exploration
   - Plot eigenvalues before/after optimization
   - Examine decay curves

## Implementation Details

### Custom Eigendecomposition

The custom gradient implementation in `my_lbfgs.py` is crucial for performance:

```python
@tf.custom_gradient
def _myeig(A):
    e, v = np.linalg.eig(A)  # Use NumPy for speed
    def grad(grad_e, grad_v):
        # Implements matrix calculus for eigendecomposition
        # See equation 4.77 in Boeddeker et al. 2017
    return (e, v), grad
```

### Numerical Stability

Several techniques ensure numerical stability:
- Reciprocal function with epsilon: `x/(x*x + ε)`
- Normalization of U and V matrices
- Large penalty for unstable eigenvalues

### Memory Efficiency

For large networks (N=500), memory usage can be significant:
- Batch processing in `train_models.py`
- Efficient tensor operations in TensorFlow
- Selective computation of required quantities

## Running the Code

### Basic Training

```bash
# Train a single model
python train_models.py -N 100 -F 0.1 -b 0.01 -f output.dill

# Parameter sweep
python train_models.py -N "100 300 500" -F "0.05 0.1 0.2" -k 5
```

### Command-line Arguments

- `-N`: Network sizes (space-separated)
- `-F`: Rank fractions (space-separated)
- `-b`: Beta values for smoothness
- `-k`: Number of random initializations
- `-T`: Time constant tau
- `-a`: Stability penalty alpha
- `-L`: L-BFGS iterations
- `-A`: ADAM epochs

### Notebook Workflow

1. Start with `Preparatory network optimization.ipynb` for understanding
2. Use `prep_opt.ipynb` for interactive experimentation
3. Run `train_models.py` for systematic parameter sweeps
4. Analyze results with statistics notebook

## Key Insights and Tips

1. **Eigenvalue visualization**: Always plot eigenvalues before/after optimization to verify stability

2. **Decay time analysis**: The notebooks compute theoretical decay times - compare with simulations

3. **Parameter sensitivity**: 
   - Small n/N: Limited control but faster optimization
   - Large n/N: Better control but risk of overfitting
   - β tuning: Start with 0.01, adjust based on output smoothness needs

4. **Convergence monitoring**: L-BFGS may converge quickly (20 iterations) or slowly (200+ iterations) depending on initialization

5. **Numerical precision**: The code uses float64 for critical computations to ensure accuracy

## References

The mathematical framework is based on:
- Linear systems theory for neural networks
- Low-rank matrix perturbation theory
- Eigenvalue optimization techniques

The custom gradient implementation follows:
- Boeddeker et al. 2017 (arXiv:1701.00392) for eigendecomposition gradients

## Future Extensions

Potential improvements and extensions:
1. Nonlinear dynamics incorporation
2. Structured connectivity constraints
3. Multiple input/output channels
4. Online learning algorithms
5. Hardware acceleration for larger networks
