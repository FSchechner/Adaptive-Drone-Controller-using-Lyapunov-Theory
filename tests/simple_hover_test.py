import numpy as np
import sys
sys.path.insert(0, '../src')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rigid_body_quadrotor import RigidBodyQuadrotor

def simple_hover_and_turn():
    """
    Simple test: Hover at 3m, turn left, turn right

    Timeline:
    0-2s:   Takeoff to 3m
    2-5s:   Hover
    5-8s:   Turn left (positive yaw torque)
    8-11s:  Turn right (negative yaw torque)
    11-15s: Hover again
    """
    print("Testing: Hover → Turn Left → Turn Right\n")

    # Create quadrotor
    quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)

    # Initial state: on ground, level
    x0 = np.zeros(12)

    # Control strategy
    def control(t):
        F_hover = quad.hover_thrust()

        if t < 2:
            # Takeoff: extra thrust
            return np.array([F_hover * 1.2, 0, 0, 0])
        elif t < 5:
            # Hover
            return np.array([F_hover, 0, 0, 0])
        elif t < 8:
            # Turn left: positive yaw torque
            return np.array([F_hover, 0, 0, 0.3])
        elif t < 11:
            # Turn right: negative yaw torque
            return np.array([F_hover, 0, 0, -0.3])
        else:
            # Hover
            return np.array([F_hover, 0, 0, 0])

    # Simulate
    def dynamics(t, x):
        u = control(t)
        return quad.dynamics(t, x, u)

    print("Running simulation...")
    sol = solve_ivp(dynamics, [0, 15], x0, dense_output=True, max_step=0.01, rtol=1e-9)

    # Sample trajectory
    t_eval = np.linspace(0, 15, 1500)
    x_history = sol.sol(t_eval).T

    # Extract states
    x = x_history[:, 0]
    y = x_history[:, 1]
    z = x_history[:, 2]
    vx = x_history[:, 3]
    vy = x_history[:, 4]
    vz = x_history[:, 5]
    phi = x_history[:, 6]
    theta = x_history[:, 7]
    psi = x_history[:, 8]
    omega_x = x_history[:, 9]
    omega_y = x_history[:, 10]
    omega_z = x_history[:, 11]

    # Print summary
    print("\nResults:")
    print(f"  Final altitude:     {z[-1]:.3f} m (target: ~3m)")
    print(f"  Final yaw angle:    {np.degrees(psi[-1]):.1f}° (should be ~0°)")
    print(f"  Max yaw rate:       {np.degrees(np.max(np.abs(omega_z))):.1f} deg/s")
    print(f"  Position drift:     x={x[-1]:.3f}m, y={y[-1]:.3f}m")
    print(f"  Final roll/pitch:   φ={np.degrees(phi[-1]):.3f}°, θ={np.degrees(theta[-1]):.3f}°")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Position
    axes[0, 0].plot(t_eval, x, 'r-', label='x')
    axes[0, 0].plot(t_eval, y, 'g-', label='y')
    axes[0, 0].plot(t_eval, z, 'b-', linewidth=2, label='z')
    axes[0, 0].axhline(3, color='b', linestyle='--', alpha=0.5, label='Target altitude')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Velocity
    axes[0, 1].plot(t_eval, vx, 'r-', label='vx')
    axes[0, 1].plot(t_eval, vy, 'g-', label='vy')
    axes[0, 1].plot(t_eval, vz, 'b-', label='vz')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Euler angles
    axes[1, 0].plot(t_eval, np.degrees(phi), 'r-', label='φ (roll)')
    axes[1, 0].plot(t_eval, np.degrees(theta), 'g-', label='θ (pitch)')
    axes[1, 0].plot(t_eval, np.degrees(psi), 'b-', linewidth=2, label='ψ (yaw)')
    axes[1, 0].set_ylabel('Angle (deg)')
    axes[1, 0].set_title('Euler Angles (Orientation)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Add phase markers
    for ax in [axes[1, 0]]:
        ax.axvspan(0, 2, alpha=0.1, color='green', label='Takeoff')
        ax.axvspan(2, 5, alpha=0.1, color='blue', label='Hover')
        ax.axvspan(5, 8, alpha=0.1, color='yellow', label='Turn Left')
        ax.axvspan(8, 11, alpha=0.1, color='orange', label='Turn Right')
        ax.axvspan(11, 15, alpha=0.1, color='blue')

    # Angular velocity
    axes[1, 1].plot(t_eval, np.degrees(omega_x), 'r-', label='ωx')
    axes[1, 1].plot(t_eval, np.degrees(omega_y), 'g-', label='ωy')
    axes[1, 1].plot(t_eval, np.degrees(omega_z), 'b-', linewidth=2, label='ωz (yaw rate)')
    axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[1, 1].set_title('Angular Velocity (Body Frame)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 3D trajectory
    ax_3d = fig.add_subplot(3, 2, 5, projection='3d')
    ax_3d.plot(x, y, z, 'b-', linewidth=2)
    ax_3d.scatter([0], [0], [0], c='g', s=100, marker='o', label='Start')
    ax_3d.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='x', label='End')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory')
    ax_3d.legend()
    ax_3d.grid(True)

    # XY plane view (top-down)
    axes[2, 1].plot(x, y, 'b-', linewidth=2)
    axes[2, 1].scatter([0], [0], c='g', s=100, marker='o', label='Start')
    axes[2, 1].scatter([x[-1]], [y[-1]], c='r', s=100, marker='x', label='End')

    # Draw orientation arrows at key times
    key_times = [2, 5, 8, 11, 14.5]
    for t_key in key_times:
        idx = np.argmin(np.abs(t_eval - t_key))
        pos_x, pos_y = x[idx], y[idx]
        yaw_angle = psi[idx]

        # Arrow pointing in yaw direction
        arrow_len = 0.3
        dx = arrow_len * np.cos(yaw_angle)
        dy = arrow_len * np.sin(yaw_angle)

        axes[2, 1].arrow(pos_x, pos_y, dx, dy,
                        head_width=0.1, head_length=0.1,
                        fc='red', ec='red', alpha=0.6)

    axes[2, 1].set_xlabel('X (m)')
    axes[2, 1].set_ylabel('Y (m)')
    axes[2, 1].set_title('Top View (arrows show yaw direction)')
    axes[2, 1].axis('equal')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('simple_hover_test.png', dpi=300)
    print(f"\nPlot saved: simple_hover_test.png")
    #plt.show()

    # Check if test passed
    print("\n" + "="*60)
    altitude_ok = abs(z[-1] - 3.0) < 0.5
    yaw_ok = abs(np.degrees(psi[-1])) < 10
    drift_ok = np.sqrt(x[-1]**2 + y[-1]**2) < 1.0

    print("TEST RESULTS:")
    print(f"  ✓ Altitude stable:     {altitude_ok} (z = {z[-1]:.3f}m)")
    print(f"  ✓ Yaw returns to 0:    {yaw_ok} (ψ = {np.degrees(psi[-1]):.1f}°)")
    print(f"  ✓ Minimal drift:       {drift_ok} (r = {np.sqrt(x[-1]**2 + y[-1]**2):.3f}m)")
    print("="*60)

    if altitude_ok and yaw_ok and drift_ok:
        print("\n✓ ALL TESTS PASSED!")
        print("Rigid body dynamics working correctly!")
    else:
        print("\n⚠ SOME ISSUES DETECTED")
        print("Check plots for details")

if __name__ == "__main__":
    simple_hover_and_turn()
