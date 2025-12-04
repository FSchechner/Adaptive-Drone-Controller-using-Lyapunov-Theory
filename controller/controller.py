import numpy as np
class QuadcopterController:
    def __init__(self, mass=1.2, g=9.81):
        self.m = mass
        self.g = g

        self.Kp_xy = 0.12
        self.Kd_xy = 0.5
        self.Kp_z = 5.0
        self.Kd_z = 3.0
        self.Kp_att = 6.0
        self.Kd_att = 0.6 
        
    def controller(self,state,target_pos): 
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        omega_x, omega_y, omega_z = state[9:12]
        
        x_d, y_d, z_d = target_pos

        distance = np.sqrt((x_d-x)**2 + (y_d-y)**2)
        eps = 1e-5

        if distance > eps:
            psi_d = np.arctan2(y_d - y, x_d - x)
            e_psi = psi_d - psi
            while e_psi > np.pi:
                e_psi -= 2*np.pi
            while e_psi < -np.pi:
                e_psi += 2*np.pi
        else:
            e_psi = 0.0

        v_forward = vx * np.cos(psi) + vy * np.sin(psi)
        theta_d = self.Kp_xy * distance - self.Kd_xy * v_forward
        phi_d = 0.0

        max_tilt = np.radians(30)
        theta_d = np.clip(theta_d, -max_tilt, max_tilt)

        az_d = self.Kp_z * (z_d - z) - self.Kd_z * vz
        F_total = self.m * (az_d + self.g)
        F_total = np.clip(F_total, 0, 20)

        tau_phi = self.Kp_att * (phi_d - phi) - self.Kd_att * omega_x
        tau_theta = self.Kp_att * (theta_d - theta) - self.Kd_att * omega_y
        tau_psi = self.Kp_att * e_psi - self.Kd_att * omega_z

        return np.array([F_total, tau_phi, tau_theta, tau_psi]) 