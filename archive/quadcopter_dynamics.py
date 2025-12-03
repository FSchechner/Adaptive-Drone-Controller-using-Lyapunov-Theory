import numpy as np

class QuadcopterDynamics:
    def __init__(self):
        self.m = 5.2
        self.g = 9.81
        self.b = 3.13e-5
        self.l = 0.32
        self.d = 7.5e-7

        self.Ixx = 3.8e-3
        self.Iyy = 3.8e-3
        self.Izz = 7.1e-3

        self.Cdx = 0.1
        self.Cdy = 0.1
        self.Cdz = 0.15

        self.Cax = 0.1
        self.Cay = 0.1
        self.Caz = 0.15

        self.Jr = 6e-5

    def state_derivative(self, t, state, motor_speeds):
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = state
        w1, w2, w3, w4 = motor_speeds

        w1_sq = w1**2
        w2_sq = w2**2
        w3_sq = w3**2
        w4_sq = w4**2

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        w_sum = w1_sq + w2_sq + w3_sq + w4_sq

        ax = (self.b / self.m) * w_sum * (cos_psi * sin_theta * cos_phi + sin_psi * sin_phi) - (self.Cdx / self.m) * vx
        ay = (self.b / self.m) * w_sum * (cos_phi * sin_psi * sin_theta - cos_psi * sin_phi) - (self.Cdy / self.m) * vy
        az = (self.b / self.m) * w_sum * cos_theta * cos_phi - (self.Cdz / self.m) * vz - self.g

        Omega_r = w1 - w2 + w3 - w4

        p_dot = (self.l * self.b * (w4_sq - w2_sq) / self.Ixx
                 - self.Cax * p**2 / self.Ixx
                 - self.Jr * Omega_r * q / self.Ixx
                 - (self.Izz - self.Iyy) * q * r / self.Ixx)

        q_dot = (self.l * self.b * (w3_sq - w1_sq) / self.Iyy
                 - self.Cay * q**2 / self.Iyy
                 + self.Jr * Omega_r * p / self.Iyy
                 - (self.Ixx - self.Izz) * p * r / self.Iyy)

        r_dot = (self.d * (w1_sq - w2_sq + w3_sq - w4_sq) / self.Izz
                 - self.Caz * r**2 / self.Izz
                 - (self.Iyy - self.Ixx) * p * q / self.Izz)

        state_dot = np.array([
            vx,
            vy,
            vz,
            p,
            q,
            r,
            ax,
            ay,
            az,
            p_dot,
            q_dot,
            r_dot
        ])

        return state_dot

    def motors_to_controls(self, motor_speeds):
        w1, w2, w3, w4 = motor_speeds
        u1 = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        u2 = self.l * self.b * (w4**2 - w2**2)
        u3 = self.l * self.b * (w3**2 - w1**2)
        u4 = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        return np.array([u1, u2, u3, u4])

    def controls_to_motors(self, controls):
        u1, u2, u3, u4 = controls

        A = np.array([
            [self.b, self.b, self.b, self.b],
            [0, -self.l * self.b, 0, self.l * self.b],
            [-self.l * self.b, 0, self.l * self.b, 0],
            [self.d, -self.d, self.d, -self.d]
        ])

        w_squared = np.linalg.solve(A, controls)
        w_squared = np.maximum(w_squared, 0)
        motor_speeds = np.sqrt(w_squared)

        return motor_speeds
