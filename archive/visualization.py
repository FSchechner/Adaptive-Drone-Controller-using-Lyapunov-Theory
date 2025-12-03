import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_results(results):
    t = results['t']
    states = results['states']
    desired_states = results['desired_states']

    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    phi, theta, psi = states[:, 3], states[:, 4], states[:, 5]

    x_d, y_d, z_d, psi_d = desired_states[:, 0], desired_states[:, 1], desired_states[:, 2], desired_states[:, 3]

    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_d, y_d, z_d, 'b-', label='Setpoint', linewidth=2)
    ax1.plot(x, y, z, 'r--', label='Real Trajectory', linewidth=2)
    ax1.set_xlabel('X axis (m)')
    ax1.set_ylabel('Y axis (m)')
    ax1.set_zlabel('Z axis (m)')
    ax1.set_title('Setpoint and Real Trajectory')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, x, 'r-', label='X')
    ax2.plot(t, x_d, 'b--', label='X Desired')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X response')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, y, 'r-', label='Y')
    ax3.plot(t, y_d, 'b--', label='Y Desired')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y response')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, z, 'r-', label='Z')
    ax4.plot(t, z_d, 'b--', label='Z Desired')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Z response')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, np.rad2deg(phi), 'r-', label='Phi')
    ax5.plot(t, np.rad2deg(theta), 'g-', label='Theta')
    ax5.plot(t, np.rad2deg(psi), 'b-', label='Psi')
    ax5.set_xlabel('time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Attitude Angles')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, x - x_d, 'r-', label='X error')
    ax6.plot(t, y - y_d, 'g-', label='Y error')
    ax6.plot(t, z - z_d, 'b-', label='Z error')
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('Error (m)')
    ax6.set_title('Position Errors')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300)
    plt.show()


def plot_detailed_response(results):
    t = results['t']
    states = results['states']
    desired_states = results['desired_states']

    x, y = states[:, 0], states[:, 1]
    x_d, y_d = desired_states[:, 0], desired_states[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, x, 'r-', label='X')
    axes[0].plot(t, x_d, 'b--', label='X Desired')
    axes[0].set_ylabel('X (m)')
    axes[0].set_title('X response')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, y, 'r-', label='Y')
    axes[1].plot(t, y_d, 'b--', label='Y Desired')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Y response')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('xy_response.png', dpi=300)
    plt.show()


def compute_performance_metrics(results):
    states = results['states']
    desired_states = results['desired_states']

    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    x_d, y_d, z_d = desired_states[:, 0], desired_states[:, 1], desired_states[:, 2]

    x_error = x - x_d
    y_error = y - y_d
    z_error = z - z_d

    x_max_error = np.max(np.abs(x_error))
    y_max_error = np.max(np.abs(y_error))
    z_max_error = np.max(np.abs(z_error))

    x_rmse = np.sqrt(np.mean(x_error**2))
    y_rmse = np.sqrt(np.mean(y_error**2))
    z_rmse = np.sqrt(np.mean(z_error**2))

    x_overshoot_idx = np.argmax(x)
    y_overshoot_idx = np.argmax(y)

    x_overshoot = (x[x_overshoot_idx] - x_d[x_overshoot_idx]) / x_d[x_overshoot_idx] * 100 if x_d[x_overshoot_idx] > 0 else 0
    y_overshoot = (y[y_overshoot_idx] - y_d[y_overshoot_idx]) / y_d[y_overshoot_idx] * 100 if y_d[y_overshoot_idx] > 0 else 0

    print("Performance Metrics:")
    print(f"X: Max error = {x_max_error:.4f} m, RMSE = {x_rmse:.4f} m, Overshoot = {x_overshoot:.2f}%")
    print(f"Y: Max error = {y_max_error:.4f} m, RMSE = {y_rmse:.4f} m, Overshoot = {y_overshoot:.2f}%")
    print(f"Z: Max error = {z_max_error:.4f} m, RMSE = {z_rmse:.4f} m")
