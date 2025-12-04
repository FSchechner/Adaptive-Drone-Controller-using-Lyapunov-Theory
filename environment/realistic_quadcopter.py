import numpy as np

class RealisticQuadcopter:
    """
    Realistic quadcopter environment mimicking real flight controller behavior.

    Features:
    - Fixed 1kHz control loop (dt = 0.001s, like real hardware)
    - Sensor noise (IMU: accelerometer, gyroscope)
    - Motor dynamics (first-order lag)
    - GPS noise and update rate (5-10Hz)
    - Euler integration (what real microcontrollers use)

    State: [x, y, z, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
    Control: [F, τφ, τθ, τψ]
    """

    def __init__(self, mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142,
                 enable_noise=True, enable_motor_dynamics=True):

        # Physical parameters
        self.m = mass
        self.g = 9.81
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.diag([1/Ixx, 1/Iyy, 1/Izz])
        self.D = np.diag([0.1, 0.1, 0.15])  # Drag coefficients

        # Control loop frequency (typical flight controller: 500-1000 Hz)
        self.dt = 0.001  # 1kHz = 1ms timestep

        # State vector
        self.state = np.zeros(12)  # [pos, vel, euler, omega]
        self.state[2] = 0.01  # Start slightly above ground

        # Motor dynamics (first-order lag: tau * F_dot + F = F_cmd)
        self.motor_time_constant = 0.02  # 20ms motor lag (realistic)
        self.F_actual = mass * self.g  # Current actual thrust
        self.tau_actual = np.zeros(3)  # Current actual torques

        # Control command
        self.u_cmd = np.array([self.m * self.g, 0, 0, 0])

        # Sensor noise parameters (based on real IMU specs)
        self.enable_noise = enable_noise
        self.accel_noise_std = 0.02  # 0.02 m/s² (typical for good IMU)
        self.gyro_noise_std = np.radians(0.1)  # 0.1 deg/s
        self.gps_noise_std = 0.5  # 0.5 m position noise
        self.gps_update_rate = 10  # Hz
        self.gps_counter = 0
        self.last_gps_pos = np.zeros(3)

        # Motor dynamics flag
        self.enable_motor_dynamics = enable_motor_dynamics

        # Time
        self.t = 0.0
        self.step_count = 0

    def step(self, u_cmd):
        """
        Step simulation forward by dt=0.001s (1ms, like real hardware).

        Parameters:
        -----------
        u_cmd : np.array (4,)
            Commanded control [F, τφ, τθ, τψ]

        Returns:
        --------
        state_measured : dict
            Measured state with sensor noise
        """
        self.u_cmd = np.array(u_cmd)

        # Apply motor dynamics (first-order filter)
        if self.enable_motor_dynamics:
            alpha = self.dt / (self.motor_time_constant + self.dt)
            self.F_actual += alpha * (u_cmd[0] - self.F_actual)
            self.tau_actual += alpha * (u_cmd[1:4] - self.tau_actual)
        else:
            self.F_actual = u_cmd[0]
            self.tau_actual = u_cmd[1:4]

        # Extract state
        pos = self.state[0:3]
        vel = self.state[3:6]
        euler = self.state[6:9]
        omega = self.state[9:12]
        phi, theta, psi = euler

        # Rotation matrix
        R = self._rotation_matrix(phi, theta, psi)

        # ===== Translational Dynamics =====
        thrust_world = R @ np.array([0, 0, self.F_actual])
        gravity_world = np.array([0, 0, -self.m * self.g])
        drag_world = -self.D @ vel

        accel = (thrust_world + gravity_world + drag_world) / self.m

        # ===== Rotational Dynamics =====
        I_omega = self.I @ omega
        gyro_term = np.cross(omega, I_omega)
        omega_dot = self.I_inv @ (self.tau_actual - gyro_term)

        # ===== Euler Angle Kinematics =====
        W = self._euler_rate_matrix(phi, theta)
        euler_dot = W @ omega

        # Euler integration (what real flight controllers do)
        self.state[0:3] += vel * self.dt  # position
        self.state[3:6] += accel * self.dt  # velocity
        self.state[6:9] += euler_dot * self.dt  # euler angles
        self.state[9:12] += omega_dot * self.dt  # angular velocity

        # Wrap angles to [-π, π]
        self.state[6:9] = np.arctan2(np.sin(self.state[6:9]),
                                      np.cos(self.state[6:9]))

        # Update time
        self.t += self.dt
        self.step_count += 1

        # Return measured state with sensor noise
        return self._measure_state()

    def _measure_state(self):
        """
        Simulate sensor measurements with realistic noise.

        Returns:
        --------
        measurements : dict
            Noisy sensor readings
        """
        if not self.enable_noise:
            return {
                'position': self.state[0:3].copy(),
                'velocity': self.state[3:6].copy(),
                'euler': self.state[6:9].copy(),
                'omega': self.state[9:12].copy(),
                'acceleration': np.zeros(3),  # Would need to compute
                't': self.t
            }

        # IMU measurements (high rate, ~1kHz)
        omega_measured = self.state[9:12] + np.random.normal(0, self.gyro_noise_std, 3)

        # Accelerometer measures specific force (not including gravity)
        phi, theta, psi = self.state[6:9]
        R = self._rotation_matrix(phi, theta, psi)
        vel = self.state[3:6]

        # True acceleration in world frame
        thrust_world = R @ np.array([0, 0, self.F_actual])
        gravity_world = np.array([0, 0, -self.m * self.g])
        drag_world = -self.D @ vel
        accel_world = (thrust_world + gravity_world + drag_world) / self.m

        # Transform to body frame and add gravity (what accelerometer measures)
        accel_body = R.T @ (accel_world - np.array([0, 0, -self.g]))
        accel_measured = accel_body + np.random.normal(0, self.accel_noise_std, 3)

        # GPS measurements (low rate, ~10Hz)
        if self.step_count % int(1000 / self.gps_update_rate) == 0:
            self.last_gps_pos = self.state[0:3] + np.random.normal(0, self.gps_noise_std, 3)

        # Euler angles from IMU integration (in practice, uses complementary filter)
        euler_measured = self.state[6:9].copy()  # Assume perfect for now

        return {
            'position': self.last_gps_pos.copy(),  # GPS position
            'velocity': self.state[3:6].copy(),  # Could add noise
            'euler': euler_measured,
            'omega': omega_measured,
            'acceleration': accel_measured,
            't': self.t,
            'true_state': self.state.copy()  # For debugging
        }

    def get_state(self):
        """Get true state (for ground truth comparison)."""
        return self.state.copy()

    def reset(self, x0=None):
        """Reset to initial state."""
        if x0 is None:
            self.state = np.zeros(12)
            self.state[2] = 0.01
        else:
            self.state = np.array(x0)

        self.F_actual = self.m * self.g
        self.tau_actual = np.zeros(3)
        self.t = 0.0
        self.step_count = 0
        self.last_gps_pos = self.state[0:3].copy()

        return self._measure_state()

    def _rotation_matrix(self, phi, theta, psi):
        """ZYX Euler rotation: body → world"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        return np.array([
            [cpsi*ctheta, cpsi*stheta*sphi - spsi*cphi, cpsi*stheta*cphi + spsi*sphi],
            [spsi*ctheta, spsi*stheta*sphi + cpsi*cphi, spsi*stheta*cphi - cpsi*sphi],
            [-stheta, ctheta*sphi, ctheta*cphi]
        ])

    def _euler_rate_matrix(self, phi, theta):
        """Convert angular velocity to Euler rates: η̇ = W·ω"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta = np.cos(theta)
        ttheta = np.tan(theta)

        if np.abs(ctheta) < 1e-3:
            raise ValueError(f"Gimbal lock at theta = {np.degrees(theta):.1f}°")

        return np.array([
            [1, sphi*ttheta, cphi*ttheta],
            [0, cphi, -sphi],
            [0, sphi/ctheta, cphi/ctheta]
        ])

    def hover_thrust(self):
        """Thrust needed to hover."""
        return self.m * self.g


if __name__ == "__main__":
    # Test realistic environment
    print("Testing Realistic Quadcopter Environment\n")

    quad = RealisticQuadcopter(mass=1.2, enable_noise=True, enable_motor_dynamics=True)

    # Hover test with sensor noise
    u_hover = np.array([quad.hover_thrust(), 0, 0, 0])

    print(f"Control loop frequency: {1/quad.dt:.0f} Hz (dt = {quad.dt*1000:.1f} ms)")
    print(f"Motor time constant: {quad.motor_time_constant*1000:.0f} ms")
    print(f"Sensor noise enabled: {quad.enable_noise}")
    print(f"GPS update rate: {quad.gps_update_rate} Hz")
    print("\nRunning 1000 steps (1 second)...")

    for i in range(1000):
        measurements = quad.step(u_hover)

    print(f"\nAfter 1 second:")
    print(f"  True position: {quad.get_state()[0:3]}")
    print(f"  GPS position: {measurements['position']}")
    print(f"  Position noise: {np.linalg.norm(quad.get_state()[0:3] - measurements['position']):.3f} m")
    print(f"  True altitude: {quad.get_state()[2]:.4f} m")
    print(f"  Gyro noise std: {np.degrees(quad.gyro_noise_std):.2f} deg/s")
    print(f"  Accel noise std: {quad.accel_noise_std:.3f} m/s²")
