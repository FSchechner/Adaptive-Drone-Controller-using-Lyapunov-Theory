import numpy as np


class AdaptiveController:
    def __init__(self,
                 m_nominal=1.9,
                 g=9.81,
                 lambda_xy=4.8221,
                 lambda_z=7.1715,
                 k_xy=15.0,
                 k_z=7.9871,
                 gamma_alpha=1.3129,
                 gamma_d=0.1,
                 alpha_min=0.2,
                 alpha_max=2.0,
                 d_max=20.0,
                 F_max=60.0):

        self.m_nominal = m_nominal
        self.g = g
        self.dt = 0.01

        self.Lambda = np.diag([lambda_xy, lambda_xy, lambda_z])
        self.K = np.diag([k_xy, k_xy, k_z])

        self.alpha_hat = 1.0 / m_nominal
        self.d_hat = np.array([0.0, 0.0, 0.0])
        self.theta_hat = np.array([self.alpha_hat, 0.0, 0.0, 0.0])
        self.theta_nominal = self.theta_hat.copy()

        self.Gamma = np.diag([gamma_alpha, gamma_d, gamma_d, gamma_d])

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.d_max = d_max

        self.F_max = F_max

        self.pos_prev = None
        self.vel_prev_d = None

    def compute_control(self, pos, vel, pos_d, vel_d=None, acc_d=None):
        # Desired velocity from finite differences if not supplied
        if vel_d is None:
            if self.pos_prev is None:
                vel_d = np.array([0.0, 0.0, 0.0])
            else:
                vel_d = (pos_d - self.pos_prev) / self.dt

        # Desired acceleration from finite differences if not supplied
        if acc_d is None:
            if self.vel_prev_d is None:
                acc_d = np.array([0.0, 0.0, 0.0])
            else:
                acc_d = (vel_d - self.vel_prev_d) / self.dt

        e_pos = pos - pos_d
        e_vel = vel - vel_d

        s = e_vel + self.Lambda @ e_pos

        a_cmd = -self.Lambda @ e_vel - self.K @ s + acc_d

        e3 = np.array([0.0, 0.0, 1.0])
        a_des = a_cmd + self.g * e3

        m_hat = 1.0 / max(self.alpha_hat, 1e-6)

        # Control law from the paper: F = m_hat * (a_des - \hat{d})
        F_control = m_hat * (a_des - self.d_hat)
        F_control = np.clip(F_control, -self.F_max, self.F_max)

        # Regressor uses the applied control force
        Y = np.zeros((3, 4))
        Y[:, 0] = F_control
        Y[:, 1:4] = np.eye(3)

        theta_dot = self.Gamma @ (Y.T @ s)

        self.theta_hat += self.dt * theta_dot

        self.theta_hat[0] = np.clip(self.theta_hat[0], self.alpha_min, self.alpha_max)
        self.theta_hat[1] = np.clip(self.theta_hat[1], -self.d_max, self.d_max)
        self.theta_hat[2] = np.clip(self.theta_hat[2], -self.d_max, self.d_max)
        self.theta_hat[3] = np.clip(self.theta_hat[3], -self.d_max, self.d_max)

        self.alpha_hat = self.theta_hat[0]
        self.d_hat = self.theta_hat[1:4]

        self.pos_prev = pos_d
        self.vel_prev_d = vel_d

        return F_control

    def reset(self):
        self.alpha_hat = 1.0 / self.m_nominal
        self.d_hat = np.array([0.0, 0.0, 0.0])
        self.theta_hat = np.array([self.alpha_hat, 0.0, 0.0, 0.0])
        self.theta_nominal = self.theta_hat.copy()
        self.pos_prev = None
        self.vel_prev_d = None

    def get_estimates(self):
        m_hat = 1.0 / max(self.alpha_hat, 1e-6)
        return {
            'm_hat': m_hat,
            'alpha_hat': self.alpha_hat,
            'd_hat': self.d_hat.copy(),
            'theta_hat': self.theta_hat.copy()
        }
