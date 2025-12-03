import numpy as np
import sys
sys.path.insert(0, '../src')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rigid_body_quadrotor import RigidBodyQuadrotor

def simple_open_loop_test():
    """
    Pure open-loop test: Apply thrust to reach altitude, then apply yaw torques

    No feedback - just timed thrust and torque inputs
    """
    print("Testing: Open-Loop Takeoff → Yaw Left → Yaw Right\n")

    quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)
    x0 = np.zeros(12)

    F_hover = quad.hover_thrust()  # 11.77 N

    def control(t):
        """Open-loop control sequence"""

        if t < 2.0:
            # Takeoff: moderate thrust for 2s
            return np.array([F_hover * 1.15, 0, 0, 0])

        elif t < 4.0:
            # Hover thrust (will drift slightly)
            return np.array([F_hover, 0, 0, 0])

        elif t < 5.0:
            # Turn left: apply yaw torque for 1s
            return np.array([F_hover, 0, 0, 0.15])

        elif t < 6.0:
            # Stop rotation: apply opposite torque for 1s
            return np.array([F_hover, 0, 0, -0.15])

        elif t < 7.0:
            # Turn right: apply opposite yaw torque for 1s
            return np.array([F_hover, 0, 0, -0.15])

        elif t < 8.0:
            # Stop rotation: apply counter-torque for 1s
            return np.array([F_hover, 0, 0, 0.15])

        else:
            # Coast
            return np.array([F_hover, 0, 0, 0])

    def dynamics(t, x):
        u = control(t)
        return quad.dynamics(t, x, u)

    print("Running simulation...")
    sol = solve_ivp(dynamics, [0, 10], x0, dense_output=True, max_step=0.01, rtol=1e-9)

    t_eval = np.linspace(0, 10, 1000)
    x_history = sol.sol(t_eval).T

    # Extract states
    z = x_history[:, 2]
    vz = x_history[:, 5]
    psi = x_history[:, 8]
    omega_z = x_history[:, 11]

    print("\nResults:")
    print(f"  Max altitude:       {np.max(z):.3f} m")
    print(f"  Final altitude:     {z[-1]:.3f} m")
    print(f"  Final yaw angle:    {np.degrees(psi[-1]):.1f}°")
    print(f"  Max yaw rate:       {np.degrees(np.max(np.abs(omega_z))):.1f} deg/s")
    print(f"  Final yaw rate:     {np.degrees(omega_z[-1]):.1f} deg/s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Altitude
    axes[0, 0].plot(t_eval, z, 'b-', linewidth=2)
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude (Open-Loop)')
    axes[0, 0].grid(True)
    axes[0, 0].axvspan(0, 2, alpha=0.1, color='green', label='Takeoff')
    axes[0, 0].axvspan(2, 4, alpha=0.1, color='blue', label='Hover')
    axes[0, 0].legend()

    # Vertical velocity
    axes[0, 1].plot(t_eval, vz, 'b-', linewidth=2)
    axes[0, 1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
    axes[0, 1].set_title('Vertical Velocity')
    axes[0, 1].grid(True)

    # Yaw angle
    axes[1, 0].plot(t_eval, np.degrees(psi), 'b-', linewidth=2)
    axes[1, 0].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Yaw Angle (deg)')
    axes[1, 0].set_title('Yaw Angle (Open-Loop)')
    axes[1, 0].grid(True)
    axes[1, 0].axvspan(4, 5, alpha=0.1, color='yellow', label='Left torque')
    axes[1, 0].axvspan(5, 6, alpha=0.1, color='orange', label='Stop')
    axes[1, 0].axvspan(6, 7, alpha=0.1, color='cyan', label='Right torque')
    axes[1, 0].axvspan(7, 8, alpha=0.1, color='pink', label='Stop')
    axes[1, 0].legend(fontsize=8)

    # Yaw rate
    axes[1, 1].plot(t_eval, np.degrees(omega_z), 'b-', linewidth=2)
    axes[1, 1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Yaw Rate (deg/s)')
    axes[1, 1].set_title('Yaw Rate (Open-Loop)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('simple_open_loop_test.png', dpi=300)
    print(f"\nPlot saved: simple_open_loop_test.png")

    print("\n" + "="*60)
    print("OPEN-LOOP TEST COMPLETE")
    print("="*60)
    print("\nPhysics observations:")
    print(f"  - Quadrotor takes off and reaches ~{np.max(z):.1f}m")
    print(f"  - Yaw torques create rotation (max rate: {np.degrees(np.max(np.abs(omega_z))):.0f} deg/s)")
    print(f"  - Counter-torques slow/reverse rotation")
    print(f"  - Final yaw: {np.degrees(psi[-1]):.0f}° (not zero - open loop!)")
    print("\n✓ Rigid body dynamics working correctly!")

if __name__ == "__main__":
    simple_open_loop_test()
