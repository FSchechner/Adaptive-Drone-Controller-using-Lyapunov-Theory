import numpy as np

class QuadcopterController:
    def __init__(self,
                 mass=1.2,
                 g=9.81,
                 Kp_long=1.0, Kd_long=0.8,
                 Kp_lat=0.6,  Kd_lat=0.6,
                 Kp_z=5.0,    Kd_z=3.0,
                 Kp_att=15.0, Kd_att=5.0,
                 max_tilt_deg=35.0,
                 F_max=35.0):
        self.m = mass
        self.g = g

        # Position gains in body-aligned frame (longitudinal, lateral)
        self.Kp_long = Kp_long
        self.Kd_long = Kd_long
        self.Kp_lat  = Kp_lat
        self.Kd_lat  = Kd_lat
        self.integrate = 0.0

        # Altitude gains
        self.Kp_z = Kp_z
        self.Kd_z = Kd_z

        # Attitude gains (used for phi, theta, psi)
        self.Kp_att = Kp_att
        self.Kd_att = Kd_att

        # Limits
        self.max_tilt = np.radians(max_tilt_deg)
        self.F_max = F_max

    def controller(self, state, target_pos):
        x, y, z       = state[0:3]
        vx, vy, vz    = state[3:6]
        phi, theta, psi = state[6:9]
        omega_x, omega_y, omega_z = state[9:12]

        x_d, y_d, z_d = target_pos

        # Horizontal position errors 
        ex = x_d - x
        ey = y_d - y

        # Heading control
        distance = np.sqrt(ex**2 + ey**2)
        eps = 1e-5

        if distance > eps:
            psi_d = np.arctan2(ey, ex)
            e_psi = psi_d - psi
            while e_psi > np.pi:
                e_psi -= 2.0 * np.pi
            while e_psi < -np.pi:
                e_psi += 2.0 * np.pi
        else:
            psi_d = psi
            e_psi = 0.0

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Longitudinal and lateral  position errors
        e_long =  cos_psi * ex + sin_psi * ey
        e_lat  = -sin_psi * ex + cos_psi * ey

        # Body-frame velocities
        v_long =  cos_psi * vx + sin_psi * vy
        v_lat  = -sin_psi * vx + cos_psi * vy
   
        theta_d = self.Kp_long * e_long - self.Kd_long * v_long

        # Roll left/right for lateral motion 
        phi_d   = -self.Kp_lat * e_lat + self.Kd_lat * v_lat

        # Tilt limits
        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d   = np.clip(phi_d,   -self.max_tilt, self.max_tilt)

        # --- Altitude control (world z) ---
        e_z = z_d - z
        az_d = self.Kp_z * e_z - self.Kd_z * vz          
        Fz   = self.m * (az_d + self.g)                 

        cos_theta = np.cos(theta)
        cos_phi   = np.cos(phi)
        denom = cos_theta * cos_phi
        denom = np.clip(denom, 0.2, 1.0)

        F_total = Fz / denom
        F_total = np.clip(F_total, 0.0, self.F_max)

        # --- Attitude control 
        tau_phi   = self.Kp_att * (phi_d   - phi)   - self.Kd_att * omega_x
        tau_theta = self.Kp_att * (theta_d - theta) - self.Kd_att * omega_y
        tau_psi   = self.Kp_att * (e_psi)           - self.Kd_att * omega_z

        return np.array([F_total, tau_phi, tau_theta, tau_psi])