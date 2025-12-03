import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0


class CascadedPIDController:
    def __init__(self, mass=5.2, gravity=9.81):
        self.m = mass
        self.g = gravity

        self.pid_x = PIDController(kp=1.0, ki=0.0, kd=2.0)
        self.pid_y = PIDController(kp=1.0, ki=0.0, kd=2.0)
        self.pid_z = PIDController(kp=10.0, ki=0.5, kd=5.0)

        self.pid_phi = PIDController(kp=4.0, ki=0.0, kd=0.8, output_limits=(-5, 5))
        self.pid_theta = PIDController(kp=4.0, ki=0.0, kd=0.8, output_limits=(-5, 5))
        self.pid_psi = PIDController(kp=3.0, ki=0.0, kd=0.5, output_limits=(-2, 2))

    def compute_controls(self, state, desired_state, dt):
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = state
        x_d, y_d, z_d, psi_d = desired_state

        e_z = z_d - z
        u1 = self.m * (self.g + self.pid_z.compute(e_z, dt))
        u1 = max(0, u1)

        e_x = x_d - x
        e_y = y_d - y

        ax_d = self.pid_x.compute(e_x, dt)
        ay_d = self.pid_y.compute(e_y, dt)

        phi_d = np.arcsin(self.m * (ax_d * np.sin(psi) - ay_d * np.cos(psi)) / u1) if u1 > 0 else 0
        theta_d = np.arcsin(self.m * (ax_d * np.cos(psi) + ay_d * np.sin(psi)) / u1) if u1 > 0 else 0

        phi_d = np.clip(phi_d, -np.pi/6, np.pi/6)
        theta_d = np.clip(theta_d, -np.pi/6, np.pi/6)

        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = psi_d - psi

        while e_psi > np.pi:
            e_psi -= 2 * np.pi
        while e_psi < -np.pi:
            e_psi += 2 * np.pi

        u2 = self.pid_phi.compute(e_phi, dt)
        u3 = self.pid_theta.compute(e_theta, dt)
        u4 = self.pid_psi.compute(e_psi, dt)

        controls = np.array([u1, u2, u3, u4])
        return controls

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_phi.reset()
        self.pid_theta.reset()
        self.pid_psi.reset()
