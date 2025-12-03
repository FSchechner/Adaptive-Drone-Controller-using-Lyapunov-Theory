import numpy as np

class RigidBodyQuadrotor:
    """
    Rigid body quadrotor dynamics

    State: x = [x,y,z, vx,vy,vz, φ,θ,ψ, ωx,ωy,ωz]
    Control: u = [F, τφ, τθ, τψ]
    """

    def __init__(self, mass=1.2, Ixx=0.0081, Iyy=0.0081, Izz=0.0142):
        self.m = mass
        self.g = 9.81
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.diag([1/Ixx, 1/Iyy, 1/Izz])
        self.D = np.diag([0.1, 0.1, 0.15])

    def dynamics(self, t, x, u):
        """
        Rigid body dynamics: ẋ = f(x, u)

        Translation: m·v̇ = R·[0,0,F] + [0,0,-mg] - D·v
        Rotation: I·ω̇ = τ - ω×(I·ω)
        """
        v = x[3:6]
        phi, theta, psi = x[6:9]
        omega = x[9:12]

        F, tau_phi, tau_theta, tau_psi = u

        R = self.rotation_matrix(phi, theta, psi)

        thrust_world = R @ np.array([0, 0, F])
        gravity_world = np.array([0, 0, -self.m * self.g])
        drag_world = -self.D @ v

        v_dot = (thrust_world + gravity_world + drag_world) / self.m

        I_omega = self.I @ omega
        gyro_term = np.cross(omega, I_omega)
        tau = np.array([tau_phi, tau_theta, tau_psi])
        omega_dot = self.I_inv @ (tau - gyro_term)

        W = self.euler_rate_matrix(phi, theta)
        eta_dot = W @ omega

        return np.concatenate([v, v_dot, eta_dot, omega_dot])

    def rotation_matrix(self, phi, theta, psi):
        """ZYX Euler rotation: body → world"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        return np.array([
            [cpsi*ctheta, cpsi*stheta*sphi - spsi*cphi, cpsi*stheta*cphi + spsi*sphi],
            [spsi*ctheta, spsi*stheta*sphi + cpsi*cphi, spsi*stheta*cphi - cpsi*sphi],
            [-stheta, ctheta*sphi, ctheta*cphi]
        ])

    def euler_rate_matrix(self, phi, theta):
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
        """Thrust needed to hover"""
        return self.m * self.g

    def get_body_axes(self, eta):
        """Get body axes in world frame for visualization"""
        R = self.rotation_matrix(*eta)
        return R[:, 0], R[:, 1], R[:, 2]

    def angular_momentum(self, omega):
        """Angular momentum: L = I·ω"""
        return self.I @ omega

    def total_energy(self, x):
        """Total energy: KE + PE"""
        v, omega = x[3:6], x[9:12]
        KE = 0.5 * self.m * np.dot(v, v) + 0.5 * omega @ self.I @ omega
        PE = self.m * self.g * x[2]
        return KE + PE
