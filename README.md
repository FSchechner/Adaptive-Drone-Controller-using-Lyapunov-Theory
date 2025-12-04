# Quadcopter Trajectory Tracking

**Rigid body quadrotor simulator with cascaded PD control**

Research project for robotics course
*Harvard University*

---

## Quick Start

```bash
# Run controller test with waypoint tracking
cd tests
python3 test_controller.py

# Run simple hover test
python3 test_simple_hover.py
```

---

## Project Structure

```
.
├── README.md                      # This file
├── src/                           # Core physics simulation
│   └── rigid_body_quadrotor.py        # 6-DOF rigid body dynamics
├── controller/                    # Control algorithms
│   └── controller.py                  # Cascaded PD controller
├── tests/                         # Test simulations
│   ├── test_controller.py             # Waypoint tracking test
│   ├── test_simple_hover.py           # Hover stability test
│   └── simple_open_loop_test.py       # Open-loop dynamics test
├── environment/                   # Alternative dynamics implementations
│   └── Quadcopter-Dynamics.py
├── docs/                          # Documentation
│   ├── RIGID_BODY_README.md           # Detailed rigid body explanation
│   └── introduction.*                 # Project introduction
└── archive/                       # Old implementations
```

---

## What is This?

This is a **complete rigid body quadrotor simulator** implementing proper 6-DOF dynamics:

### State (12D)
```
x = [x, y, z, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
```

### Key Physics
1. **Translation**: `m·v̇ = R·[0,0,F] + [0,0,-mg] - D·v`
   Thrust only acts along body z-axis!

2. **Rotation**: `I·ω̇ = τ - ω×(I·ω)`
   Gyroscopic term couples the axes!

3. **Orientation**: `η̇ = W(φ,θ)·ω`
   Converts body angular velocity to Euler rates

---

## Basic Usage

```python
from src.rigid_body_quadrotor import RigidBodyQuadrotor
import numpy as np
from scipy.integrate import solve_ivp

# Create quadrotor
quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)

# Initial state: [x,y,z, vx,vy,vz, φ,θ,ψ, ωx,ωy,ωz]
x0 = np.zeros(12)
x0[2] = 3.0  # Start at 3m altitude

# Control: [F, τφ, τθ, τψ]
u_hover = np.array([quad.hover_thrust(), 0, 0, 0])

# Simulate
def dynamics(t, x):
    return quad.dynamics(t, x, u_hover)

sol = solve_ivp(dynamics, [0, 10], x0, dense_output=True)
```

---

## Control Law

The controller uses a cascaded architecture with outer loop (position) → inner loop (attitude).

### Outer Loop: Position Control

**Yaw Control** (point at target):
```
ψ_d = arctan2(y_d - y, x_d - x)  if distance > ε
e_ψ = wrap_to_pi(ψ_d - ψ)
```

**XY Position** (car-like control):
```
distance = √[(x_d - x)² + (y_d - y)²]
v_forward = vx·cos(ψ) + vy·sin(ψ)
θ_d = Kp_xy·distance - Kd_xy·v_forward
θ_d = clip(θ_d, -30°, +30°)
```

**Z Position** (altitude control):
```
az_d = Kp_z·(z_d - z) - Kd_z·vz
F = m·(az_d + g)
F = clip(F, 0, 20N)
```

### Inner Loop: Attitude Control

**PD control on angles**:
```
τ_φ = Kp_att·(φ_d - φ) - Kd_att·ωx
τ_θ = Kp_att·(θ_d - θ) - Kd_att·ωy
τ_ψ = Kp_att·(e_ψ) - Kd_att·ωz
```

### Control Output
```
u = [F, τ_φ, τ_θ, τ_ψ]
```

**Default Gains**:
- Position: `Kp_xy=0.12, Kd_xy=0.5, Kp_z=5.0, Kd_z=3.0`
- Attitude: `Kp_att=3.0, Kd_att=0.6`

---

## Why Rigid Body?

**Point mass model (WRONG)**:
- Can generate force in any direction
- No gyroscopic effects
- Orientation doesn't matter

**Rigid body model (CORRECT)**:
- Thrust ONLY along body z-axis
- Gyroscopic coupling: `ω×(I·ω)`
- Must tilt to move horizontally
- Inertia tensor affects rotation

See [docs/RIGID_BODY_README.md](docs/RIGID_BODY_README.md) for detailed physics explanation.

---

## Tests

Controller tests verify:
- ✓ Hover stability at target altitude
- ✓ Waypoint tracking in 3D space
- ✓ Yaw pointing toward target (car-like behavior)
- ✓ Smooth trajectory interpolation
- ✓ Open-loop dynamics validation

Run: `python3 tests/test_controller.py`

---

## Key Features

### Simple, Clean Code
- ~95 lines of core dynamics
- Clear physics implementation
- Easy to extend

### Proper Physics
- Full rigid body dynamics
- Gyroscopic effects
- Frame transformations
- Conservation laws

### Comprehensive Testing
- 7 physics validation tests
- Energy/momentum checks
- Numerical accuracy verification

### Educational Demos
- 3D trajectory visualization
- Gyroscopic effect demonstration
- Point mass vs rigid body comparison

---

## Potential Extensions

Possible future improvements:

1. **Trajectory optimization** - Minimum-time or minimum-energy paths
2. **Gain tuning** - Systematic PD gain selection
3. **State estimation** - Add noise, implement Kalman filter
4. **Hardware implementation** - Deploy on actual quadrotor

---

## References

- **Bouabdallah (2007)**: *Design and Control of Quadrotors with Focus on the Factors Influencing the Quadrotor Design*
- **Abdelhay & Zakriti (2019)**: *Modeling of a Quadcopter Trajectory Tracking System Using PID Controller*
- **Beard & McLain (2012)**: *Small Unmanned Aircraft: Theory and Practice*

---

## Contact

GitHub: [Adaptive-Drone-Controller-using-Contraction-Theory](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory)

---

**Remember**: A quadrotor is a rigid body, not a point mass. It can only push along its body z-axis!
