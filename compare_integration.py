import numpy as np
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'environment')
from rigid_body_quadrotor import RigidBodyQuadrotor
from scipy.integrate import solve_ivp

# Test energy conservation over 10 seconds
quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)

# Initial state: hovering at 3m
x0 = np.zeros(12)
x0[2] = 3.0

# Control: perfect hover thrust
u = np.array([quad.hover_thrust(), 0, 0, 0])

# Solve with RK45 (adaptive)
def dynamics(t, x):
    return quad.dynamics(t, x, u)

sol = solve_ivp(dynamics, [0, 10], x0, method='RK45', rtol=1e-9)
E0_rk45 = quad.total_energy(sol.y[:, 0])
Ef_rk45 = quad.total_energy(sol.y[:, -1])

print("Energy Conservation Test (10s hover):")
print(f"RK45 (adaptive):  E0 = {E0_rk45:.6f} J, Ef = {Ef_rk45:.6f} J, ΔE = {Ef_rk45-E0_rk45:.6f} J")

# Euler with dt=0.01
x = x0.copy()
t = 0
dt = 0.01
E0_euler_01 = quad.total_energy(x)
while t < 10:
    x = x + quad.dynamics(t, x, u) * dt
    t += dt
Ef_euler_01 = quad.total_energy(x)
print(f"Euler dt=0.01:    E0 = {E0_euler_01:.6f} J, Ef = {Ef_euler_01:.6f} J, ΔE = {Ef_euler_01-E0_euler_01:.6f} J")

# Euler with dt=0.001
x = x0.copy()
t = 0
dt = 0.001
E0_euler_001 = quad.total_energy(x)
while t < 10:
    x = x + quad.dynamics(t, x, u) * dt
    t += dt
Ef_euler_001 = quad.total_energy(x)
print(f"Euler dt=0.001:   E0 = {E0_euler_001:.6f} J, Ef = {Ef_euler_001:.6f} J, ΔE = {Ef_euler_001-E0_euler_001:.6f} J")
