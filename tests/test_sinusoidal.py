import numpy as np
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../controller')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rigid_body_quadrotor import RigidBodyQuadrotor
from controller import QuadcopterController

def test_sinusoidal_trajectory():
    print("Testing Sinusoidal Trajectory Tracking\n")

    quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)
    controller = QuadcopterController(mass=1.2, g=9.81)

    x0 = np.zeros(12)
    x0[2] = 0.1

    def get_target(t):
        forward_speed = 5
        amplitude = 2.0
        frequency = 0.05
        z_altitude = 3.0

        x_d = forward_speed * t
        y_d = amplitude * np.sin(2 * np.pi * frequency * t)
        z_d = z_altitude

        return np.array([x_d, y_d, z_d])

    def dynamics(t, x):
        target = get_target(t)
        u = controller.controller(x, target)
        return quad.dynamics(t, x, u)

    print("Running simulation...")
    t_final = 20
    sol = solve_ivp(dynamics, [0, t_final], x0, dense_output=True, max_step=0.01, rtol=1e-9)

    t_eval = np.linspace(0, t_final, 2000)
    x_history = sol.sol(t_eval).T

    x = x_history[:, 0]
    y = x_history[:, 1]
    z = x_history[:, 2]
    vx = x_history[:, 3]
    vy = x_history[:, 4]
    vz = x_history[:, 5]
    phi = x_history[:, 6]
    theta = x_history[:, 7]
    psi = x_history[:, 8]
    omega_z = x_history[:, 11]

    targets = np.array([get_target(t) for t in t_eval])
    x_d = targets[:, 0]
    y_d = targets[:, 1]
    z_d = targets[:, 2]

    print("\nResults:")
    print(f"  Final position:     ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")
    print(f"  Target position:    ({x_d[-1]:.2f}, {y_d[-1]:.2f}, {z_d[-1]:.2f})")

    error_x = x - x_d
    error_y = y - y_d
    error_z = z - z_d
    tracking_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

    print(f"  Mean tracking error: {np.mean(tracking_error[1000:]):.3f} m")
    print(f"  Max tracking error:  {np.max(tracking_error):.3f} m")
    print(f"  RMS tracking error:  {np.sqrt(np.mean(tracking_error[1000:]**2)):.3f} m")

    fig = plt.figure(figsize=(15, 12))

    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    ax1.plot(x_d, y_d, z_d, 'b--', linewidth=2, label='Desired', alpha=0.6)
    ax1.plot(x, y, z, 'r-', linewidth=2, label='Actual')
    ax1.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(x, y, 'r-', linewidth=2, label='Actual')
    ax2.plot(x_d, y_d, 'b--', linewidth=2, label='Desired', alpha=0.6)

    for i in range(0, len(t_eval), 200):
        yaw_angle = psi[i]
        arrow_len = 0.3
        dx = arrow_len * np.cos(yaw_angle)
        dy = arrow_len * np.sin(yaw_angle)
        ax2.arrow(x[i], y[i], dx, dy, head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.5)

    ax2.scatter([x[0]], [y[0]], c='g', s=100, marker='o', label='Start')
    ax2.scatter([x[-1]], [y[-1]], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (arrows show heading)')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(t_eval, x, 'r-', label='x')
    ax3.plot(t_eval, x_d, 'r--', alpha=0.5)
    ax3.plot(t_eval, y, 'g-', label='y')
    ax3.plot(t_eval, y_d, 'g--', alpha=0.5)
    ax3.plot(t_eval, z, 'b-', label='z')
    ax3.plot(t_eval, z_d, 'b--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(t_eval, tracking_error, 'b-', linewidth=2)
    ax4.axhline(np.mean(tracking_error[1000:]), color='r', linestyle='--',
                label=f'Mean: {np.mean(tracking_error[1000:]):.3f}m')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Tracking Error (m)')
    ax4.set_title('Total Tracking Error')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(t_eval, error_x, 'r-', label='x error')
    ax5.plot(t_eval, error_y, 'g-', label='y error')
    ax5.plot(t_eval, error_z, 'b-', label='z error')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Error (m)')
    ax5.set_title('Component-wise Tracking Error')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(t_eval, np.degrees(phi), 'r-', label='φ (roll)')
    ax6.plot(t_eval, np.degrees(theta), 'g-', label='θ (pitch)')
    ax6.plot(t_eval, np.degrees(psi), 'b-', label='ψ (yaw)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Angle (deg)')
    ax6.set_title('Euler Angles')
    ax6.legend()
    ax6.grid(True)

    ax7 = fig.add_subplot(3, 3, 7)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    ax7.plot(t_eval, speed, 'b-', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Speed (m/s)')
    ax7.set_title('Total Speed')
    ax7.grid(True)

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(t_eval, vx, 'r-', label='vx')
    ax8.plot(t_eval, vy, 'g-', label='vy')
    ax8.plot(t_eval, vz, 'b-', label='vz')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Velocity (m/s)')
    ax8.set_title('Velocity Components')
    ax8.legend()
    ax8.grid(True)

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(t_eval, np.degrees(omega_z), 'b-', linewidth=2)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Yaw Rate (deg/s)')
    ax9.set_title('Angular Velocity (Yaw)')
    ax9.grid(True)

    plt.tight_layout()
    plt.savefig('test_sinusoidal.png', dpi=300)
    print(f"\nPlot saved: test_sinusoidal.png")

    print("\n" + "="*60)
    mean_error = np.mean(tracking_error[1000:])
    max_error = np.max(tracking_error)
    rms_error = np.sqrt(np.mean(tracking_error[1000:]**2))

    print("SINUSOIDAL TRAJECTORY TEST RESULTS:")
    print(f"  Mean tracking error:  {mean_error:.3f} m")
    print(f"  RMS tracking error:   {rms_error:.3f} m")
    print(f"  Max tracking error:   {max_error:.3f} m")
    print(f"  Forward speed:        0.8 m/s")
    print(f"  Lateral amplitude:    2.0 m")
    print(f"  Wave frequency:       0.5 Hz (period: 2.0 s)")
    print("="*60)

    if mean_error < 0.5:
        print("\n✓ SINUSOIDAL TRAJECTORY TEST PASSED!")
        print("Successfully tracked forward sinusoidal path!")
    else:
        print("\n⚠ Controller needs tuning for better tracking")

if __name__ == "__main__":
    test_sinusoidal_trajectory()
