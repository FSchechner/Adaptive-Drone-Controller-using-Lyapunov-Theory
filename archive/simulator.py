import numpy as np
from scipy.integrate import solve_ivp
from quadcopter_dynamics import QuadcopterDynamics
from pid_controller import CascadedPIDController

class QuadcopterSimulator:
    def __init__(self):
        self.dynamics = QuadcopterDynamics()
        self.controller = CascadedPIDController(mass=self.dynamics.m, gravity=self.dynamics.g)

    def simulate(self, trajectory, initial_state, t_span, dt=0.01):
        t_eval = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t_eval)

        states = np.zeros((n_steps, 12))
        controls = np.zeros((n_steps, 4))
        motor_speeds = np.zeros((n_steps, 4))
        desired_states = np.zeros((n_steps, 4))

        states[0] = initial_state
        current_state = initial_state.copy()

        self.controller.reset()

        for i in range(1, n_steps):
            t = t_eval[i-1]

            desired_state = trajectory(t)
            desired_states[i-1] = desired_state

            control = self.controller.compute_controls(current_state, desired_state, dt)
            controls[i-1] = control

            motors = self.dynamics.controls_to_motors(control)
            motor_speeds[i-1] = motors

            def state_derivative_wrapper(t, state):
                return self.dynamics.state_derivative(t, state, motors)

            sol = solve_ivp(
                state_derivative_wrapper,
                [t, t + dt],
                current_state,
                method='RK45',
                dense_output=True
            )

            current_state = sol.y[:, -1]
            states[i] = current_state

        desired_states[-1] = trajectory(t_eval[-1])
        controls[-1] = controls[-2]
        motor_speeds[-1] = motor_speeds[-2]

        results = {
            't': t_eval,
            'states': states,
            'controls': controls,
            'motor_speeds': motor_speeds,
            'desired_states': desired_states
        }

        return results
