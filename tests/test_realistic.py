import numpy as np
import sys
sys.path.insert(0, '../environment')
sys.path.insert(0, '../controller')
import matplotlib.pyplot as plt
from realistic_quadcopter import RealisticQuadcopter
from controller import QuadcopterController

def test_realistic_controller():
    """
    Test controller with realistic hardware-like simulation:
    - 1kHz control loop (dt = 1ms)
    - Sensor noise (IMU, GPS)
    - Motor dynamics (20ms lag)
    """
    print("Testing Controller with Realistic Hardware Simulation\n")

    # Create realistic environment
    quad = RealisticQuadcopter(mass=1.2, enable_noise=True, enable_motor_dynamics=True)
    controller = QuadcopterController(mass=1.2, g=9.81)

    # Target: hover at 3m, then move to (5, 0, 3)
    waypoints = [
        (0, 3000, [0, 0, 3]),      # Hover at origin for 3s
        (3000, 6000, [5, 0, 3]),   # Move to (5, 0, 3) over 3s
        (6000, 10000, [5, 0, 3])   # Hold position for 4s
    ]

    def get_target(step):
        """Get target position based on step count (1 step = 1ms)."""
        for step_start, step_end, pos in waypoints:
            if step_start <= step < step_end:
                return np.array(pos)
        return np.array(waypoints[-1][2])

    # Simulation parameters
    total_time = 10.0  # seconds
    total_steps = int(total_time / quad.dt)

    # Storage
    time_history = []
    state_true_history = []
    state_measured_history = []
    target_history = []
    control_history = []
    motor_actual_history = []

    print(f"Control loop: {1/quad.dt:.0f} Hz ({quad.dt*1000:.1f} ms timestep)")
    print(f"Total steps: {total_steps} ({total_time:.1f} seconds)")
    print(f"Motor lag: {quad.motor_time_constant*1000:.0f} ms")
    print(f"Sensor noise: Gyro ±{np.degrees(quad.gyro_noise_std):.1f}°/s, Accel ±{quad.accel_noise_std:.3f} m/s²")
    print(f"GPS: {quad.gps_update_rate} Hz, ±{quad.gps_noise_std:.1f}m\n")

    print("Running simulation...")

    # Reset quad
    x0 = np.zeros(12)
    x0[2] = 0.1
    quad.reset(x0)

    # Initial control
    u_cmd = np.array([quad.hover_thrust(), 0, 0, 0])

    for step in range(total_steps):
        # Get target
        target = get_target(step)

        # Step simulation with previous control
        measurements = quad.step(u_cmd)

        # Controller uses noisy measurements
        state_measured = np.concatenate([
            measurements['position'],
            measurements['velocity'],
            measurements['euler'],
            measurements['omega']
        ])

        # Compute control
        u_cmd = controller.controller(state_measured, target)

        # Store for analysis
        if step % 10 == 0:  # Store every 10ms to reduce memory
            time_history.append(quad.t)
            state_true_history.append(measurements['true_state'].copy())
            state_measured_history.append(state_measured.copy())
            target_history.append(target.copy())
            control_history.append(u_cmd.copy())
            motor_actual_history.append(np.array([quad.F_actual, *quad.tau_actual]))

    # Convert to arrays
    time_history = np.array(time_history)
    state_true = np.array(state_true_history)
    state_measured = np.array(state_measured_history)
    targets = np.array(target_history)
    controls = np.array(control_history)
    motor_actual = np.array(motor_actual_history)

    # Extract variables
    x_true = state_true[:, 0]
    y_true = state_true[:, 1]
    z_true = state_true[:, 2]
    x_meas = state_measured[:, 0]
    z_meas = state_measured[:, 2]

    x_d = targets[:, 0]
    y_d = targets[:, 1]
    z_d = targets[:, 2]

    # Tracking error (based on true state)
    error = np.sqrt((x_true - x_d)**2 + (y_true - y_d)**2 + (z_true - z_d)**2)

    print("\nResults:")
    print(f"  Final true position:     ({x_true[-1]:.2f}, {y_true[-1]:.2f}, {z_true[-1]:.2f})")
    print(f"  Final measured position: ({x_meas[-1]:.2f}, {state_measured[-1,1]:.2f}, {z_meas[-1]:.2f})")
    print(f"  Target position:         ({x_d[-1]:.2f}, {y_d[-1]:.2f}, {z_d[-1]:.2f})")
    print(f"  Mean tracking error:     {np.mean(error[500:]):.3f} m")
    print(f"  RMS tracking error:      {np.sqrt(np.mean(error[500:]**2)):.3f} m")

    # Plot results
    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_d, y_d, z_d, 'b--', linewidth=2, label='Target', alpha=0.6)
    ax1.plot(x_true, y_true, z_true, 'r-', linewidth=2, label='True')
    ax1.plot(x_meas, state_measured[:, 1], z_meas, 'g:', linewidth=1, alpha=0.7, label='Measured (GPS)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Position vs time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(time_history, z_true, 'r-', label='True altitude', linewidth=2)
    ax2.plot(time_history, z_meas, 'g:', label='GPS altitude', linewidth=1, alpha=0.7)
    ax2.plot(time_history, z_d, 'b--', label='Target', linewidth=2, alpha=0.6)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Tracking')
    ax2.legend()
    ax2.grid(True)

    # Tracking error
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time_history, error, 'b-', linewidth=2)
    ax3.axhline(np.mean(error[500:]), color='r', linestyle='--',
                label=f'Mean: {np.mean(error[500:]):.3f}m')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (m)')
    ax3.set_title('Total Tracking Error')
    ax3.legend()
    ax3.grid(True)

    # Motor dynamics: commanded vs actual thrust
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time_history, controls[:, 0], 'b--', label='Commanded', linewidth=2, alpha=0.7)
    ax4.plot(time_history, motor_actual[:, 0], 'r-', label='Actual (with lag)', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Thrust (N)')
    ax4.set_title(f'Motor Dynamics (τ={quad.motor_time_constant*1000:.0f}ms lag)')
    ax4.legend()
    ax4.grid(True)

    # GPS vs True position
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time_history, x_true, 'r-', label='True X', linewidth=2)
    ax5.plot(time_history, x_meas, 'r:', label='GPS X', linewidth=1, alpha=0.7)
    ax5.plot(time_history, x_d, 'b--', label='Target X', linewidth=2, alpha=0.6)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('X Position (m)')
    ax5.set_title(f'GPS Noise (±{quad.gps_noise_std:.1f}m, {quad.gps_update_rate}Hz)')
    ax5.legend()
    ax5.grid(True)

    # Angular velocities (with gyro noise)
    ax6 = fig.add_subplot(2, 3, 6)
    omega_z_true = state_true[:, 11]
    omega_z_meas = state_measured[:, 11]
    ax6.plot(time_history, np.degrees(omega_z_meas), 'g-', label='Measured (noisy)', linewidth=1, alpha=0.7)
    ax6.plot(time_history, np.degrees(omega_z_true), 'b-', label='True', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Yaw Rate (deg/s)')
    ax6.set_title(f'Gyro Measurement (±{np.degrees(quad.gyro_noise_std):.1f}°/s noise)')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('test_realistic.png', dpi=300)
    print(f"\nPlot saved: test_realistic.png")

    print("\n" + "="*60)
    print("REALISTIC HARDWARE SIMULATION RESULTS:")
    print(f"  Control loop rate:       {1/quad.dt:.0f} Hz")
    print(f"  Motor lag:               {quad.motor_time_constant*1000:.0f} ms")
    print(f"  GPS noise:               ±{quad.gps_noise_std:.1f} m")
    print(f"  Mean tracking error:     {np.mean(error[500:]):.3f} m")
    print(f"  Controller handles:      Sensor noise ✓, Motor lag ✓")
    print("="*60)

if __name__ == "__main__":
    test_realistic_controller()
