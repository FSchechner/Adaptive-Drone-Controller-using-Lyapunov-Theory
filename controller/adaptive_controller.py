import numpy as np
import sys
sys.path.insert(0,'../Drone')
from Drone_1 import Drone

class QuadcopterController:
    def __init__(self,
                 g=9.81,
                 Kp_long=1.6517, Kd_long=1.4134,
                 Kp_lat=0.2020,  Kd_lat=0.9168,
                 Kp_z=4.2928,    Kd_z=1.2188,
                 Kp_att=15.9895, Kd_att=2.0513,
                 max_tilt_deg=35.0,
                 gamma_m=0.0420,
                 gamma_I=7.3491):
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
        self.I = np.array([self.Drone.Ixx, self.Drone.Iyy, self.Drone.Izz])

        self.a = np.array([
            1.0/self.m_true,
            self.Drone.Ixx,
            self.Drone.Iyy,
            self.Drone.Izz
        ])

        self.Gamma = np.diag([gamma_m, gamma_I, gamma_I, gamma_I])

        self.a_min = np.array([
            1.0/(2.5*self.m_true),
            0.3*self.Drone.Ixx,
            0.3*self.Drone.Iyy,
            0.3*self.Drone.Izz
        ])
        self.a_max = np.array([
            1.0/(0.5*self.m_true),
            3.0*self.Drone.Ixx,
            3.0*self.Drone.Iyy,
            3.0*self.Drone.Izz
        ])

        self.m_hat = 1.0 / self.a[0]
        self.I_hat = self.a[1:4].copy()

        self.estimate_history = {
            'time': [],
            'm_hat': [],
            'I_hat_x': [],
            'I_hat_y': [],
            'I_hat_z': []
        }

        self.pos_history = np.array([0.0, 0.0, 0.0])
        self.psi_d_fixed = None

    def rotation_matrix_to_euler_zyx(self, R):
        """
        Extract Euler angles (roll, pitch, yaw) from rotation matrix.
        Uses ZYX convention: R = Rz(psi) * Ry(theta) * Rx(phi)
        Handles singularities at pitch ≈ ±90°
        """
        # Check for singularity
        sin_theta = -R[2, 0]
        sin_theta = np.clip(sin_theta, -1.0, 1.0)

        theta = np.arcsin(sin_theta)

        # Check if we're near singularity (cos(theta) ≈ 0)
        if np.abs(np.cos(theta)) > 1e-6:
            # Normal case
            phi = np.arctan2(R[2, 1], R[2, 2])
            psi = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Singularity: theta ≈ ±90°
            # Set phi = 0 and solve for psi
            phi = 0.0
            if sin_theta > 0:  # theta ≈ +90°
                psi = np.arctan2(-R[0, 1], R[1, 1])
            else:  # theta ≈ -90°
                psi = np.arctan2(R[0, 1], R[1, 1])

        return phi, theta, psi

    def controller(self, state, target_pos, dt=0.01):
        """
        Unified Nominal Controller based on the paper.

        Position Control: Computes desired force F_des
        Attitude Determination: Constructs R_d and extracts Euler angles
        Attitude Control: PD control with gyroscopic compensation
        """
        # Extract state
        pos = state[0:3]
        vel = state[3:6]
        phi, theta, psi = state[6:9]
        omega = state[9:12]

        # Initialize on first call
        if self.psi_d_fixed is None:
            self.psi_d_fixed = psi
            self.pos_history = target_pos.copy()

        # Desired velocity (numerical differentiation)
        vel_d = (target_pos - self.pos_history) / dt

        # === STEP 1: POSITION CONTROL ===
        # Paper equation: F_des = m(K_p(ξ_d - ξ) + K_d(ξ̇_d - ξ̇) + g)
        # where g = [0, 0, 9.81]^T (corrected from scalar)

        pos_error = target_pos - pos
        vel_error = vel_d - vel

        # Desired acceleration in world frame
        acc_d = np.array([
            self.Kp_x * pos_error[0] + self.Kd_x * vel_error[0],
            self.Kp_y * pos_error[1] + self.Kd_y * vel_error[1],
            self.Kp_z * pos_error[2] + self.Kd_z * vel_error[2]
        ])

        # Desired force vector (add gravity as vector [0, 0, g])
        F_des = self.m_hat * (acc_d + np.array([0, 0, self.g]))

        # === STEP 2: ATTITUDE DETERMINATION ===
        # Extract desired angles from F_des direction

        # Desired thrust direction (third column of R_d)
        F_des_norm = np.linalg.norm(F_des)
        if F_des_norm > 0.1:
            z_b_d = F_des / F_des_norm
        else:
            z_b_d = np.array([0.0, 0.0, 1.0])

        # Construct R_d with thrust direction and fixed yaw (resolves ambiguity)
        # Method: Use desired yaw to define intermediate x-axis direction
        x_c = np.array([np.cos(self.psi_d_fixed), np.sin(self.psi_d_fixed), 0.0])

        # y_b_d = z_b_d × x_c (perpendicular to both)
        y_b_d = np.cross(z_b_d, x_c)
        y_b_d_norm = np.linalg.norm(y_b_d)

        if y_b_d_norm > 1e-6:
            y_b_d = y_b_d / y_b_d_norm
        else:
            # Singularity: z_b_d aligned with x_c
            # Use perpendicular vector in xy-plane
            y_b_d = np.array([-np.sin(self.psi_d_fixed), np.cos(self.psi_d_fixed), 0.0])

        # x_b_d = y_b_d × z_b_d (complete right-handed frame)
        x_b_d = np.cross(y_b_d, z_b_d)

        # Construct rotation matrix R_d = [x_b_d | y_b_d | z_b_d]
        R_d = np.column_stack([x_b_d, y_b_d, z_b_d])

        # Extract desired Euler angles from R_d
        phi_d, theta_d, psi_d = self.rotation_matrix_to_euler_zyx(R_d)

        # Limit tilt angles
        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d = np.clip(phi_d, -self.max_tilt, self.max_tilt)

        # === STEP 2b: RECOMPUTE THRUST ===
        # After clipping angles, recompute F to match gravity
        # F_z_world = F * cos(theta) * cos(phi) must equal m*(az_d + g)
        az_total = acc_d[2] + self.g
        cos_theta_d = np.cos(theta_d)
        cos_phi_d = np.cos(phi_d)
        denom = cos_theta_d * cos_phi_d
        denom = np.clip(denom, 0.1, 1.0)  # Prevent division by zero

        F = self.m_hat * az_total / denom
        F = np.clip(F, 0.0, self.F_max)

        # === STEP 3: ATTITUDE CONTROL ===
        # Paper: τ = Î(K_p,att(η_d - η) + K_d,att(ω_d - ω)) + ω × (Îω)

        # Attitude errors
        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = self.psi_d_fixed - psi

        # Wrap yaw error to [-π, π]
        while e_psi > np.pi:
            e_psi -= 2.0 * np.pi
        while e_psi < -np.pi:
            e_psi += 2.0 * np.pi

        att_error = np.array([e_phi, e_theta, e_psi])

        # Desired angular velocity (zero for position tracking)
        omega_d = np.zeros(3)
        omega_error = omega_d - omega

        # PD control term: Î(K_p * e_att + K_d * e_omega)
        tau_pd = self.I_hat * (self.Kp_att * att_error + self.Kd_att * omega_error)

        # Gyroscopic compensation term: ω × (Îω)
        I_omega = self.I_hat * omega
        gyro_term = np.cross(omega, I_omega)

        # Total torque
        tau = tau_pd + gyro_term

        # Saturate torques
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        # === STEP 4: ADAPTATION LAW ===
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        R = np.array([
            [c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi],
            [s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi],
            [-s_theta, c_theta*s_phi, c_theta*c_phi]
        ])

        e3 = np.array([0, 0, 1])
        Re3F = R @ e3 * F

        Y_m = np.dot(Re3F - vel, vel_error)

        omega_error = omega_d - omega
        Y_I = np.abs(omega_error)

        a_dot = np.zeros(4)
        a_dot[0] = -self.Gamma[0, 0] * Y_m
        a_dot[1:4] = -np.diag(self.Gamma)[1:4] * Y_I * omega_error

        self.a += a_dot * dt
        self.a = np.clip(self.a, self.a_min, self.a_max)

        self.m_hat = 1.0 / self.a[0]
        self.I_hat = self.a[1:4]

        # Update position history
        self.pos_history = target_pos.copy()

        return np.array([F, tau[0], tau[1], tau[2]])

    def record_estimates(self, t):
        self.estimate_history['time'].append(t)
        self.estimate_history['m_hat'].append(self.m_hat)
        self.estimate_history['I_hat_x'].append(self.I_hat[0])
        self.estimate_history['I_hat_y'].append(self.I_hat[1])
        self.estimate_history['I_hat_z'].append(self.I_hat[2])

    def get_estimates(self):
        return {
            'm_hat': self.m_hat,
            'm_true': self.m_true,
            'm_error_pct': abs(self.m_hat - self.m_true) / self.m_true * 100,
            'I_hat': self.I_hat.copy(),
            'I_true': self.I.copy(),
            'a': self.a.copy()
        }

    def plot_estimates(self, save_path='adaptation_results.png'):
        import matplotlib.pyplot as plt
        import os

        if len(self.estimate_history['time']) == 0:
            print("No estimate history to plot. Call record_estimates(t) during simulation.")
            return

        t = np.array(self.estimate_history['time'])
        m_hat = np.array(self.estimate_history['m_hat'])
        I_hat_x = np.array(self.estimate_history['I_hat_x'])
        I_hat_y = np.array(self.estimate_history['I_hat_y'])
        I_hat_z = np.array(self.estimate_history['I_hat_z'])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(t, m_hat, 'b-', linewidth=2, label='$\hat{m}$ (estimated)')
        axes[0].axhline(y=self.m_true, color='r', linestyle='--', linewidth=2, label=f'$m_{{true}}$ = {self.m_true:.2f} kg')
        axes[0].set_xlabel('Time [s]', fontsize=12)
        axes[0].set_ylabel('Mass [kg]', fontsize=12)
        axes[0].set_title('Mass Adaptation', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=11)

        axes[1].plot(t, I_hat_x, 'r-', linewidth=2, label=f'$\hat{{I}}_{{xx}}$ (estimated)')
        axes[1].plot(t, I_hat_y, 'g-', linewidth=2, label=f'$\hat{{I}}_{{yy}}$ (estimated)')
        axes[1].plot(t, I_hat_z, 'b-', linewidth=2, label=f'$\hat{{I}}_{{zz}}$ (estimated)')
        axes[1].axhline(y=self.I[0], color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'$I_{{xx,true}}$ = {self.I[0]:.4f}')
        axes[1].axhline(y=self.I[1], color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'$I_{{yy,true}}$ = {self.I[1]:.4f}')
        axes[1].axhline(y=self.I[2], color='b', linestyle='--', linewidth=1.5, alpha=0.7, label=f'$I_{{zz,true}}$ = {self.I[2]:.4f}')
        axes[1].set_xlabel('Time [s]', fontsize=12)
        axes[1].set_ylabel('Inertia [kg·m²]', fontsize=12)
        axes[1].set_title('Inertia Adaptation', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10, ncol=2)

        plt.tight_layout()

        abs_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAdaptation plot saved: {abs_path}")
        plt.close(fig)