# Rigid Body Quadrotor Simulator

## Overview

This is a **complete rigid body dynamics simulator** for quadrotors, properly modeling the vehicle as a 3D object flying through space—not a point mass with orientation tacked on as an afterthought.

## What Makes This a Rigid Body?

### Point Mass Model (WRONG for Quadrotors)
```python
# Point mass: orientation doesn't matter for forces
m·ẍ = F_thrust + F_gravity  # Force in any direction
# No inertia, no gyroscopic effects, no coupling
```

### Rigid Body Model (CORRECT)
```python
# Translation: thrust direction depends on orientation
m·v̇ = R(φ,θ,ψ)·[0,0,F] + [0,0,-m·g] - D·v
#     └─────┬─────┘
#     Thrust ONLY along body z-axis!

# Rotation: includes gyroscopic coupling
I·ω̇ = τ - ω×(I·ω)
#         └───┬───┘
#    Gyroscopic term!
```

**Key difference**: A rigid body CANNOT generate force in arbitrary directions. It can only push along its body z-axis. To move sideways, it must tilt!

## State Representation (12 DOF)

```
x = [p, v, η, ω]ᵀ ∈ ℝ¹²

where:
  p = [x, y, z]ᵀ         : position in world frame (m)
  v = [vx, vy, vz]ᵀ      : linear velocity in world frame (m/s)
  η = [φ, θ, ψ]ᵀ         : Euler angles (rad)
  ω = [ωx, ωy, ωz]ᵀ      : angular velocity in BODY frame (rad/s)
```

**Why ω is in body frame**: The inertia tensor I is constant in the body frame. In the world frame, I would be time-varying!

## The Physics

### 1. Translational Dynamics (Newton)

```
m·v̇ = F_total

F_total = F_thrust + F_gravity + F_drag

where:
  F_thrust = R(φ,θ,ψ)·[0, 0, F]ᵀ    (thrust in body z, rotated to world)
  F_gravity = [0, 0, -m·g]ᵀ          (always world -z)
  F_drag = -D·v                       (opposes velocity)
```

**Physical insight**: The rotation matrix R transforms the thrust from body frame (where it's [0,0,F]) to world frame. The 3rd column of R is the thrust direction!

### 2. Rotational Dynamics (Euler's Equation)

```
I·ω̇ + ω×(I·ω) = τ

Rearranged:
I·ω̇ = τ - ω×(I·ω)
```

For diagonal inertia:
```
Ixx·ω̇x = τx + (Iyy - Izz)·ωy·ωz
Iyy·ω̇y = τy + (Izz - Ixx)·ωz·ωx
Izz·ω̇z = τz + (Ixx - Iyy)·ωx·ωy
         └──────────┬──────────┘
           Gyroscopic coupling!
```

**Physical insight**: When spinning about two axes, conservation of angular momentum creates torque about the third axis. Spin a bicycle wheel and try to tilt it—it resists!

### 3. Orientation Kinematics

```
η̇ = W(φ,θ)·ω

where W(φ,θ) = [1,  sin(φ)tan(θ),  cos(φ)tan(θ) ]
                [0,  cos(φ),        -sin(φ)       ]
                [0,  sin(φ)/cos(θ), cos(φ)/cos(θ) ]
```

This converts body angular velocity to Euler angle rates.

**Warning**: Singularity at θ = ±90° (gimbal lock)!

### 4. The Rotation Matrix R(φ,θ,ψ)

ZYX Euler convention (aerospace standard):
```
R = Rz(ψ)·Ry(θ)·Rx(φ)

  = [cψcθ,  cψsθsφ-sψcφ,  cψsθcφ+sψsφ]
    [sψcθ,  sψsθsφ+cψcφ,  sψsθcφ-cψsφ]
    [-sθ,   cθsφ,         cθcφ       ]
```

**Properties**:
- Orthogonal: Rᵀ·R = I
- Determinant: det(R) = 1
- Inverse: R⁻¹ = Rᵀ

**Physical meaning**:
- Columns = body axes in world frame
- R·v transforms vector from body to world
- Rᵀ·v transforms vector from world to body

## Rigid Body Properties

### Mass (scalar)
```python
m = 1.2  # kg
```

### Inertia Tensor (body frame)
```python
I = [Ixx  Ixy  Ixz]
    [Ixy  Iyy  Iyz]  kg·m²
    [Ixz  Iyz  Izz]
```

For symmetric quadrotor (X or + config):
- `Ixx ≈ Iyy` (symmetry about z-axis)
- `Izz > Ixx` (more inertia about vertical)
- `Ixy = Ixz = Iyz = 0` (products zero)

**Physical meaning**:
- Large Ixx → hard to roll
- Large Iyy → hard to pitch
- Large Izz → hard to yaw

Default values:
```python
Ixx = 0.0081 kg·m²
Iyy = 0.0081 kg·m²
Izz = 0.0142 kg·m²
```

## Files

### `rigid_body_quadrotor.py`
Complete rigid body simulator implementing:
- Full 12-state nonlinear dynamics
- Proper frame transformations
- Gyroscopic effects
- Energy/momentum computation

### `test_rigid_body.py`
Comprehensive validation tests:
1. **Rotation matrix properties** (orthogonality, det=1)
2. **Free fall** (zero thrust → falls at g)
3. **Hover** (thrust=mg → stationary)
4. **Pure rotation** (torque with hover → spins in place)
5. **Gyroscopic effect** (demonstrates ω×(I·ω))
6. **Angular momentum conservation** (zero torque → L constant)
7. **Energy conservation** (no drag → E constant)

### `example_rigid_body_sim.py`
Demonstrations:
1. **Step response** with 3D visualization
2. **Gyroscopic coupling** (zero torque, but ωx changes!)
3. **Point mass vs rigid body** (shows tilt causes drift)

## Usage

### Basic Simulation
```python
from rigid_body_quadrotor import RigidBodyQuadrotor
import numpy as np
from scipy.integrate import solve_ivp

# Create quadrotor
quad = RigidBodyQuadrotor()

# Initial state: [x,y,z, vx,vy,vz, φ,θ,ψ, ωx,ωy,ωz]
x0 = np.zeros(12)
x0[2] = 3.0  # Start at 3m altitude

# Control: [F, τφ, τθ, τψ]
u = np.array([quad.m * quad.g, 0, 0, 0])  # Hover

# Simulate
def dynamics(t, x):
    return quad.dynamics(t, x, u)

sol = solve_ivp(dynamics, [0, 10], x0, dense_output=True)
```

### Run Tests
```bash
python3 test_rigid_body.py
```

Expected output:
```
✓ PASS  | Rotation Matrix Properties
✓ PASS  | Free Fall
✓ PASS  | Hover
✓ PASS  | Pure Rotation
✓ PASS  | Gyroscopic Effect
✓ PASS  | Angular Momentum Conservation
✓ PASS  | Energy Conservation

Total: 7/7 tests passed
```

### Run Demonstrations
```bash
python3 example_rigid_body_sim.py
```

Generates:
- `rigid_body_simulation.png` (3D trajectory with body frames)
- `gyroscopic_effect.png` (demonstrates ω×(I·ω) coupling)
- `rigid_body_vs_point_mass.png` (shows why orientation matters)

## Key Insights

### 1. Thrust Direction Matters
A tilted quadrotor with thrust F generates:
```
F_world = R·[0, 0, F]

For φ=30° roll tilt:
  F_x = F·sin(φ) ≈ 0.5·F  (sideways force!)
  F_z = F·cos(φ) ≈ 0.87·F (less than full thrust!)
```

If F = m·g but tilted 30°, the quadrotor FALLS because F_z < m·g!

### 2. Gyroscopic Coupling
Spinning about two axes creates torque about the third:
```
With ωy ≠ 0, ωz ≠ 0, zero torque:
  ω̇x = (Iyy - Izz)·ωy·ωz / Ixx ≠ 0
```

Even with ZERO applied torque, the angular velocity changes! This is pure rigid body physics.

### 3. Frame Consistency
- **Position p, velocity v**: world frame
- **Angular velocity ω**: body frame
- **Thrust [0,0,F]**: body frame
- **Gravity [0,0,-mg]**: world frame
- **Inertia I**: body frame (constant)

Always transform between frames using R!

### 4. Conservation Laws
With zero external forces/torques:
- **Linear momentum**: p_dot = m·v conserved
- **Angular momentum**: L = I·ω conserved (body frame)
- **Energy**: KE + PE conserved (no drag)

The tests verify these hold numerically!

## Differences from Point Mass

| Property | Point Mass | Rigid Body |
|----------|------------|------------|
| State dimension | 6 (p, v) | 12 (p, v, η, ω) |
| Force direction | Any | Only body z-axis |
| Inertia | None | I (3×3 tensor) |
| Gyroscopic effects | No | Yes (ω×(I·ω)) |
| Tilt causes drift | No | Yes! |
| Angular momentum | N/A | Conserved |
| Orientation affects thrust | No | Yes (via R) |

**Bottom line**: A quadrotor is a 3D object that can only push along one axis. Understanding this is crucial for control design!

## Mathematical Guarantees

The implementation guarantees:
1. **R is always orthogonal**: Rᵀ·R = I ✓
2. **R has determinant 1**: det(R) = 1 ✓
3. **Angular momentum conserved** (zero torque) ✓
4. **Energy conserved** (no drag) ✓
5. **Gimbal lock detected**: warns at θ ≈ ±90° ✓

## Extensions

### Use Quaternions (No Gimbal Lock)
For aggressive maneuvers with θ near ±90°, replace Euler angles with quaternions:
```python
# State: [p, v, q, ω] where q ∈ ℍ (unit quaternion)
# Kinematics: q̇ = (1/2)·q ⊗ [0, ω]
```

### Add Motor Dynamics
```python
# τ = K_motor·(ω_des² - ω_motor²)
# Adds first-order lag to control response
```

### Add Aerodynamic Effects
```python
# Blade flapping, induced drag, ground effect, etc.
```

## References

1. **Bouabdallah, S. (2007)**: "Design and Control of Quadrotors"
   - Chapter 2: Rigid body formulation
2. **Craig, J.**: "Introduction to Robotics"
   - Chapter 2: Rotation matrices
3. **Goldstein, H.**: "Classical Mechanics"
   - Chapter 5: Rigid body dynamics
4. **Mahony et al. (2012)**: "Multirotor Aerial Vehicles"
   - IEEE Control Systems Magazine

## License

MIT License - Free to use for research and education

## Contact

For questions about rigid body dynamics or implementation details, see the code comments or raise an issue.

---

**Remember**: A quadrotor is not a point—it's a rigid body that can only push along its body z-axis. Everything else follows from this constraint!
