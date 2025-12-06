import numpy as np
import sys
sys.path.insert(0,'../Drone')
from Drone_1 import Drone

class QuadcopterController:
    def __init__(self, g=9.81,
                 Kp_long=1.6517, Kd_long=1.4134,
                 Kp_lat=0.2020, Kd_lat=0.9168,
                 Kp_z=4.2928, Kd_z=1.2188,
                 Kp_att=15.9895, Kd_att=2.0513,
                 max_tilt_deg=35.0,
                 gamma_m=0.5, gamma_I=0.1):

        self.Drone = Drone()
        self.m_true = self.Drone.m
        self.g = g

        self.Kp_x = Kp_long
        self.Kd_x = Kd_long
        self.Kp_y = Kp_lat
        self.Kd_y = Kd_lat
        self.Kp_z = Kp_z
        self.Kd_z = Kd_z
        self.Kp_att = Kp_att
        self.Kd_att = Kd_att

        self.max_tilt = np.radians(max_tilt_deg)
        self.F_max = self.Drone.F_max
        self.tau_max = self.Drone.tau_max

        self.m_hat = self.m_true
        self.I_hat = np.array([self.Drone.Ixx, self.Drone.Iyy, self.Drone.Izz])

        self.gamma_m = gamma_m
        self.gamma_I = gamma_I * np.ones(3)

        self.m_min = 0.5 * self.m_true
        self.m_max = 3.0 * self.m_true
        self.I_min = 0.3 * self.I_hat.copy()
        self.I_max = 3.0 * self.I_hat.copy()

        self.pos_history = np.array([0.0, 0.0, 0.0])
        self.psi_d_fixed = None

    def rotation_matrix_to_euler_zyx(self, R):
        sin_theta = np.clip(-R[2, 0], -1.0, 1.0)
        theta = np.arcsin(sin_theta)

        if np.abs(np.cos(theta)) > 1e-6:
            phi = np.arctan2(R[2, 1], R[2, 2])
            psi = np.arctan2(R[1, 0], R[0, 0])
        else:
            phi = 0.0
            psi = np.arctan2(-R[0, 1], R[1, 1]) if sin_theta > 0 else np.arctan2(R[0, 1], R[1, 1])

        return phi, theta, psi

    def adapt_parameters(self, acc_d, vel_error, omega_error, dt):
        phi_m = np.linalg.norm(acc_d + np.array([0, 0, self.g]))
        e_v = np.linalg.norm(vel_error)

        self.m_hat += -self.gamma_m * phi_m * e_v * dt
        self.m_hat = np.clip(self.m_hat, self.m_min, self.m_max)

        for i in range(3):
            phi_I = abs(omega_error[i])
            self.I_hat[i] += -self.gamma_I[i] * phi_I * abs(omega_error[i]) * dt
            self.I_hat[i] = np.clip(self.I_hat[i], self.I_min[i], self.I_max[i])

    def controller(self, state, target_pos, dt=0.01):
        pos, vel = state[0:3], state[3:6]
        phi, theta, psi = state[6:9]
        omega = state[9:12]

        if self.psi_d_fixed is None:
            self.psi_d_fixed = psi
            self.pos_history = target_pos.copy()

        vel_d = (target_pos - self.pos_history) / dt
        pos_error = target_pos - pos
        vel_error = vel_d - vel

        acc_d = np.array([
            self.Kp_x * pos_error[0] + self.Kd_x * vel_error[0],
            self.Kp_y * pos_error[1] + self.Kd_y * vel_error[1],
            self.Kp_z * pos_error[2] + self.Kd_z * vel_error[2]
        ])

        omega_d = np.zeros(3)
        omega_error = omega_d - omega
        self.adapt_parameters(acc_d, vel_error, omega_error, dt)

        F_des = self.m_hat * (acc_d + np.array([0, 0, self.g]))
        F_des_norm = np.linalg.norm(F_des)

        if F_des_norm > 0.1:
            z_b_d = F_des / F_des_norm
        else:
            z_b_d = np.array([0.0, 0.0, 1.0])

        x_c = np.array([np.cos(self.psi_d_fixed), np.sin(self.psi_d_fixed), 0.0])
        y_b_d = np.cross(z_b_d, x_c)
        y_b_d_norm = np.linalg.norm(y_b_d)

        if y_b_d_norm > 1e-6:
            y_b_d /= y_b_d_norm
        else:
            y_b_d = np.array([-np.sin(self.psi_d_fixed), np.cos(self.psi_d_fixed), 0.0])

        x_b_d = np.cross(y_b_d, z_b_d)
        R_d = np.column_stack([x_b_d, y_b_d, z_b_d])
        phi_d, theta_d, psi_d = self.rotation_matrix_to_euler_zyx(R_d)

        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d = np.clip(phi_d, -self.max_tilt, self.max_tilt)

        az_total = acc_d[2] + self.g
        denom = np.clip(np.cos(theta_d) * np.cos(phi_d), 0.1, 1.0)
        F = np.clip(self.m_hat * az_total / denom, 0.0, self.F_max)

        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = self.psi_d_fixed - psi

        while e_psi > np.pi: e_psi -= 2.0 * np.pi
        while e_psi < -np.pi: e_psi += 2.0 * np.pi

        att_error = np.array([e_phi, e_theta, e_psi])
        tau_pd = self.I_hat * (self.Kp_att * att_error + self.Kd_att * omega_error)
        gyro_term = np.cross(omega, self.I_hat * omega)
        tau = np.clip(tau_pd + gyro_term, -self.tau_max, self.tau_max)

        self.pos_history = target_pos.copy()

        return np.array([F, tau[0], tau[1], tau[2]])

    def get_estimates(self):
        return {
            'm_hat': self.m_hat,
            'm_true': self.m_true,
            'I_hat': self.I_hat.copy()
        }
