import numpy as np
class PDController:
    def __init__(self,
                 m_nominal=1.9,
                 g=9.81,
                 Kp_xy=29.83,  # Optimized for simplified dynamics
                 Kd_xy=8.78,
                 Kp_z=11.12,
                 Ki_xy = 0.0,
                 Ki_z = 0.0,
                 Kd_z=13.81):
    
        self.m_nominal = m_nominal
        self.g = g
        # Gains
        self.Kp = np.array([Kp_xy,Kp_xy, Kp_z])
        self.Kd = np.array([Kd_xy,Kd_xy, Kd_z])
        self.Ki = np.array([Ki_xy,Ki_xy, Ki_z])
        self.integrate = np.array([0.0, 0.0, 0.0])
        self.max_integrate = 10.0
        self.dt = 0.01
        self.pos_prev = None
        self.F_max = 60

    def compute_control(self, pos, vel, pos_d):
        if self.pos_prev is None:
            vel_d = np.array([0.0,0.0,0.0])
        else:
            vel_d = (pos_d - self.pos_prev)/self.dt
        
        # Position and velocity errors
        e_pos = pos - pos_d
        e_vel = vel - vel_d
        self.integrate += e_pos *self.dt
        self.integrate = np.clip(self.integrate,-self.max_integrate, self.max_integrate)

        # PD control law
        a_des = -self.Kp * e_pos - self.Kd * e_vel - self.Ki * self.integrate

        # Gravity compensation
        e3 = np.array([0.0, 0.0, 1.0])
        a_total = a_des + self.g * e3 

        # Control force
        F_control = self.m_nominal * a_total
        self.pos_prev = pos_d
        F_control = np.clip(F_control, -self.F_max, self.F_max)
        return F_control
