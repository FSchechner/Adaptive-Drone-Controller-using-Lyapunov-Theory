import numpy as np
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../controller')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rigid_body_quadrotor import RigidBodyQuadrotor
from controller import QuadcopterController

def test_simple_hover():
    print("Testing Simple Hover\n")

    quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)
    controller = QuadcopterController(mass=1.2, g=9.81)

    x0 = np.zeros(12)
    x0[2] = 0.1

    target = [0, 0, 3]

    def dynamics(t, x):
        u = controller.controller(x, target)
        print(f"t={t:.2f}, z={x[2]:.3f}, F={u[0]:.2f}, tau_theta={u[2]:.3f}")
        return quad.dynamics(t, x, u)

    print("Running simulation...")
    sol = solve_ivp(dynamics, [0, 10], x0, dense_output=True, max_step=0.02, rtol=1e-6)

    t_eval = np.linspace(0, 10, 500)
    x_history = sol.sol(t_eval).T

    z = x_history[:, 2]
    theta = x_history[:, 7]

    print(f"\nFinal altitude: {z[-1]:.3f} m (target: 3.0 m)")
    print(f"Max altitude: {np.max(z):.3f} m")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(t_eval, z, 'b-', linewidth=2)
    axes[0].axhline(3, color='r', linestyle='--', label='Target')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].set_title('Hover Test')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_eval, np.degrees(theta), 'g-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Pitch (deg)')
    axes[1].set_title('Pitch Angle')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('test_simple_hover.png', dpi=300)
    print(f"\nPlot saved")

if __name__ == "__main__":
    test_simple_hover()
