import numpy as np
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../controller')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rigid_body_quadrotor import RigidBodyQuadrotor
from controller import QuadcopterController

def test_controller():
    print("Testing Controller + Rigid Body Quadrotor\n")

    quad = RigidBodyQuadrotor(mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142)
    controller = QuadcopterController(mass=1.2, g=9.81)

    x0 = np.zeros(12)
    x0[2] = 0.1

    waypoints = [
        [0, 0, 3],
        [5, 0, 3],
        [5, 5, 3],
        [0, 0, 3]
    ]
    waypoint_times = [0, 10, 20, 30]

    def get_target(t):
        for i in range(len(waypoint_times) - 1):
            if waypoint_times[i] <= t < waypoint_times[i+1]:
                t0, t1 = waypoint_times[i], waypoint_times[i+1]
                p0, p1 = np.array(waypoints[i]), np.array(waypoints[i+1])
                alpha = (t - t0) / (t1 - t0)
                return p0 + alpha * (p1 - p0)
        return waypoints[-1]

    def dynamics(t, x):
        target = get_target(t)
        u = controller.controller(x, target)
        return quad.dynamics(t, x, u)

    print("Running simulation...")
    sol = solve_ivp(dynamics, [0, 30], x0, dense_output=True, max_step=0.01, rtol=1e-9)

    t_eval = np.linspace(0, 30, 3000)
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
    print(f"  Final error:        {np.sqrt((x[-1]-x_d[-1])**2 + (y[-1]-y_d[-1])**2 + (z[-1]-z_d[-1])**2):.3f} m")
    print(f"  Max altitude:       {np.max(z):.2f} m")

    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_d, y_d, z_d, 'b--', linewidth=2, label='Target', alpha=0.6)
    ax1.plot(x, y, z, 'r-', linewidth=2, label='Actual')
    ax1.scatter([0], [0], [0], c='g', s=100, marker='o', label='Start')
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_eval, x, 'r-', label='x')
    ax2.plot(t_eval, x_d, 'r--', alpha=0.5)
    ax2.plot(t_eval, y, 'g-', label='y')
    ax2.plot(t_eval, y_d, 'g--', alpha=0.5)
    ax2.plot(t_eval, z, 'b-', label='z')
    ax2.plot(t_eval, z_d, 'b--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_eval, np.sqrt(vx**2 + vy**2 + vz**2), 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Total Speed')
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t_eval, np.degrees(phi), 'r-', label='φ (roll)')
    ax4.plot(t_eval, np.degrees(theta), 'g-', label='θ (pitch)')
    ax4.plot(t_eval, np.degrees(psi), 'b-', label='ψ (yaw)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Euler Angles')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x, y, 'r-', linewidth=2, label='Actual')
    ax5.plot(x_d, y_d, 'b--', linewidth=2, label='Target', alpha=0.6)

    for i in range(0, len(t_eval), 250):
        yaw_angle = psi[i]
        arrow_len = 0.5
        dx = arrow_len * np.cos(yaw_angle)
        dy = arrow_len * np.sin(yaw_angle)
        ax5.arrow(x[i], y[i], dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)

    ax5.scatter([0], [0], c='g', s=100, marker='o', label='Start')
    ax5.scatter([x[-1]], [y[-1]], c='r', s=100, marker='x', label='End')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Top View (arrows = heading)')
    ax5.axis('equal')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    error_x = x - x_d
    error_y = y - y_d
    error_z = z - z_d
    ax6.plot(t_eval, error_x, 'r-', label='x error')
    ax6.plot(t_eval, error_y, 'g-', label='y error')
    ax6.plot(t_eval, error_z, 'b-', label='z error')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Error (m)')
    ax6.set_title('Tracking Error')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('test_controller.png', dpi=300)
    print(f"\nPlot saved: test_controller.png")

    print("\n" + "="*60)
    final_error = np.sqrt((x[-1]-x_d[-1])**2 + (y[-1]-y_d[-1])**2 + (z[-1]-z_d[-1])**2)
    max_error = np.max(np.sqrt(error_x**2 + error_y**2 + error_z**2))

    print("CONTROLLER TEST RESULTS:")
    print(f"  Final tracking error:  {final_error:.3f} m")
    print(f"  Max tracking error:    {max_error:.3f} m")
    print(f"  Completed waypoints:   {len(waypoints)}")
    print("="*60)

    if final_error < 0.5:
        print("\n✓ CONTROLLER TEST PASSED!")
        print("Successfully tracked waypoints!")
    else:
        print("\n⚠ Controller needs tuning")

if __name__ == "__main__":
    test_controller()
