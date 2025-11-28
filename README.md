# Adaptive Quadrotor Control via Contraction Theory

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory)

**Research project for Prof. Jean-Jacques Slotine's robotics course**
*MIT Nonlinear Systems Laboratory*

A practical implementation of contraction-based adaptive control for quadrotor trajectory tracking under parametric uncertainty. This project demonstrates how contraction theory provides a simpler, more modular alternative to classical adaptive control design for robotic systems.

---

## Table of Contents

- [Motivation](#motivation)
- [Why Adaptive Control for Drones?](#why-adaptive-control-for-drones)
- [Theory: Contraction in 5 Minutes](#theory-contraction-in-5-minutes)
- [Contraction vs Classical Lyapunov](#contraction-vs-classical-lyapunov)
- [System Model](#system-model)
- [Controller Design](#controller-design)
- [Quick Start](#quick-start)
- [References](#references)

---

## Motivation

**The Problem:** Quadrotors must track trajectories precisely, but their parameters change during flight:

- **Payload variations:** ±67% mass change when picking up/dropping objects
- **Altitude effects:** Drag coefficients vary ±30% with air density changes

**Traditional controllers fail** when parameters deviate significantly from nominal values.

**Our solution:** Adaptive control using **contraction theory** — a framework that:
- Estimates parameters online (mass, drag)
- Guarantees exponential convergence to desired trajectory
- Adapts to parameter variations without manual retuning

---

## Why Adaptive Control for Drones?

### Real-World Scenarios

1. **Package Delivery:** Drone picks up packages → mass changes significantly → fixed controller performance degrades
2. **High-Altitude Operations:** Flying at different altitudes → air density changes → drag coefficient varies
3. **Variable Payloads:** Different package weights during delivery missions → continuous mass variations

### The Challenge

Traditional gain-scheduling requires:
- Pre-computing gains for many operating points
- Smooth interpolation between gain sets
- Conservative tuning for worst-case scenarios

Adaptive control learns parameters in real-time, maintaining performance across all conditions.

---

## Theory: Contraction in 5 Minutes

### Intuitive Explanation

**Classical stability** (Lyapunov): "All trajectories converge to an equilibrium point"

**Contraction**: "All trajectories converge to **each other** exponentially"

Imagine a rubber sheet being stretched and pulled. A **contracting system** is like a sheet that always shrinks back — no matter where two points start, the distance between them decreases exponentially.

### Mathematical Formulation

For a nonlinear system **ẋ = f(x, t)**, we measure distances using a Riemannian metric **M(x, t)**. The system is contracting if:

```
Distance between nearby trajectories: δ(t) = ||δx||_M = √(δx^T M δx)

Contraction condition: d/dt(||δx||_M) ≤ -λ||δx||_M

This implies: ||δx(t)||_M ≤ e^(-λt) ||δx(0)||_M
```

**Key insight:** Instead of computing this directly, check the matrix condition:

```
A^T M + MA + Ṁ ≺ -2λM
```

where **A = ∂f/∂x** is the Jacobian. If this holds ∀x, the system is contracting.

### Why This Matters for Control

1. **Exponential tracking:** If closed-loop system is contracting, tracking error **e(t)** → 0 exponentially
2. **Robustness:** Contraction property is preserved under matched uncertainty
3. **Modularity:** Add new parameters without redesigning controller

---

## Contraction vs Classical Lyapunov

Traditional Lyapunov-based adaptive control (backstepping) requires:
- Recursive design through position → velocity → attitude → angular rate
- 10-15 pages of algebra to derive control laws and prove stability
- Strict-feedback structure required
- Complete redesign when adding new uncertain parameters

**Contraction approach:**
1. Design nominal controller (PD + feedforward)
2. Add adaptive compensation: u = u_nominal + u_adaptive
3. Verify contraction: check eigenvalues of (A^T M + MA + Ṁ) ≺ -2λM
4. Done!

**Key advantages:** Direct verification (no recursive derivation), works for any stabilizable system, adding parameters = appending to regressor matrix

---

## System Model

### Simplified 2D Quadrotor

State vector: **x = [z, v_z, θ, ω]^T**
- **z**: Vertical position (m)
- **v_z**: Vertical velocity (m/s)
- **θ**: Pitch angle (rad)
- **ω**: Angular rate (rad/s)

### Dynamics

```
m·z̈ = u - mg - d·ż        (vertical dynamics with uncertain m, d)
I·θ̈ = τ                    (rotational dynamics, I assumed known)
```

Control inputs:
- **u**: Total thrust (N)
- **τ**: Pitch torque (N·m)

### Uncertain Parameters

**θ = [m, d]^T**
- **m**: Mass (kg) — varies with payload changes
- **d**: Drag coefficient (N·s/m) — varies with altitude/air density

### Key Property: Matched Uncertainty

All uncertainties act through the **same channels as control** (thrust and torque). This is crucial — contraction-based adaptation only works for matched uncertainty.

```
ẋ = f₀(x) + B(x)[u + φ(x)^T θ]
      ↑           ↑
   nominal    matched uncertainty
```

---

## Controller Design

### Three-Layer Architecture

```
┌─────────────────────────────────────┐
│  Trajectory Generator               │  → x_d(t), ẋ_d(t), ẍ_d(t)
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Nominal Contracting Controller     │  → u_nominal = m̂(ẍ_d + k_v·ė + k_p·e + g)
│  + Adaptive Compensation            │  → u_adaptive = -φ(x)^T·θ̂
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  Parameter Adaptation               │  → θ̂̇ = -Γ·φ(x)·B^T·M·e_v
└─────────────────────────────────────┘
```

### 1. Nominal Controller (PD + Feedforward)

Makes the **nominal system** (θ = θ_nominal) contracting:

```python
e = x - x_d              # Position error
e_v = ẋ - ẋ_d            # Velocity error

# Vertical thrust control (adaptive for mass and drag)
u_nominal = m_nominal * (ẍ_d + k_v·ė_v + k_p·e) + m_nominal·g + d_nominal·ż

# Attitude control (non-adaptive, I is known)
τ_nominal = I * (ω̇_d + k_θ·θ̃ + k_ω·ω̃)
```

**Design criterion:** Choose gains **k_p, k_v, k_θ, k_ω** such that:
```
A_nominal^T M + M A_nominal ≺ -2λM
```

### 2. Adaptive Compensation

Compensates for parameter errors using **regressor matrix**:

```python
# Regressor: φ(x, ẋ_d, ẍ_d) such that uncertainty = φ^T·θ̃
φ_z = [ẍ_d + g, ż]^T           # For vertical dynamics: [mass term, drag term]

u_adaptive = -φ_z^T · θ̂
```

**Key advantage:** To add a new uncertain parameter (e.g., propeller efficiency), just add a column to **φ**!

### 3. Adaptation Law

Standard gradient descent on parameter estimates:

```python
θ̂̇ = -Γ · φ(x) · B^T · M · e_v
```

- **Γ**: Adaptation gain matrix (diagonal, tunable)
- **B**: Control input matrix
- **M**: Contraction metric (from verification)
- **e_v**: Velocity tracking error

**Intuition:** When tracking error is large and regressor is active, update parameters quickly.

### Contraction Verification

```python
import numpy as np
from scipy.linalg import eig

# Compute Jacobian of closed-loop system
A_cl = compute_jacobian(x, x_d, θ̂)

# Check contraction condition
contraction_matrix = A_cl.T @ M + M @ A_cl + M_dot
eigenvalues = eig(contraction_matrix)[0]

is_contracting = np.all(np.real(eigenvalues) < -2*lambda_min)
```

---

## Quick Start

### Basic Simulation

```bash
# Run nominal scenario (no parameter variation)
python simulate.py --scenario nominal --duration 30

# Run with heavy payload
python simulate.py --scenario heavy_payload --duration 30

# Run with high altitude (low air density)
python simulate.py --scenario high_altitude --duration 30

# Compare fixed vs adaptive controller
python simulate.py --scenario comparison --controller both
```

### Example Code

```python
from simulation.simulator import QuadrotorSimulator
from control.adaptive_controller import AdaptiveController
from simulation.trajectories import CircleTrajectory

# Create simulator with uncertain parameters
sim = QuadrotorSimulator(
    true_mass=2.0,        # True mass (unknown to controller)
    true_drag=0.3,        # True drag coefficient
)

# Create adaptive controller with nominal estimates
controller = AdaptiveController(
    mass_nominal=1.5,     # Initial estimate
    drag_nominal=0.2,     # Initial drag estimate
    adaptation_gains=[1.0, 0.5]  # Gains for mass and drag
)

# Define trajectory
trajectory = CircleTrajectory(radius=2.0, period=10.0)

# Run simulation
results = sim.run(
    controller=controller,
    trajectory=trajectory,
    duration=30.0,
    dt=0.01
)

# Plot results
results.plot_tracking()
results.plot_parameters()
results.print_metrics()
```

### Visualization

```bash
# Generate plots for all scenarios
python analysis/plot_results.py --output figures/

# Interactive 3D visualization
python analysis/visualize_3d.py --scenario heavy_payload

# Parameter convergence analysis
python analysis/parameter_analysis.py
```

---

## References

### Primary Reference
- **Lopez, B. T., & Slotine, J. J. E.** (2020). *Contraction Metrics in Adaptive Nonlinear Control*.
  arXiv:1912.13138. [https://arxiv.org/abs/1912.13138](https://arxiv.org/abs/1912.13138)

### Foundational Papers
- **Lohmiller, W., & Slotine, J. J. E.** (1998). *On Contraction Analysis for Nonlinear Systems*.
  Automatica, 34(6), 683-696.

- **Jouffroy, J., & Slotine, J. J. E.** (2004). *Methodological Remarks on Contraction Theory*.
  IEEE CDC, 2004.

### Quadrotor Control
- **Lee, T., Leok, M., & McClamroch, N. H.** (2010). *Geometric Tracking Control of a Quadrotor UAV on SE(3)*.
  IEEE CDC, 2010.

- **Mellinger, D., & Kumar, V.** (2011). *Minimum Snap Trajectory Generation and Control for Quadrotors*.
  ICRA, 2011.

### Additional Resources
- **MIT Nonlinear Systems Laboratory:** [http://web.mit.edu/nsl/www/](http://web.mit.edu/nsl/www/)
- **Slotine's Lecture Notes:** *Applied Nonlinear Control* (Slotine & Li, 1991)
- **Contraction Theory Tutorial:** [https://arxiv.org/abs/2104.02943](https://arxiv.org/abs/2104.02943)

---

## Acknowledgments

- **Prof. Jean-Jacques Slotine** for foundational work on contraction theory
- **MIT Nonlinear Systems Laboratory** for research support
- **Lopez & Slotine (2020)** for the adaptive control framework

---

## Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory/issues](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory/issues)
- GitHub Repository: [https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory)

