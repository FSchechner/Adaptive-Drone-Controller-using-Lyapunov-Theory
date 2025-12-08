import numpy as np
from Drone import Drone, Drone_with_Package
class SimpleQuadcopter:


    def __init__(self,
                 drone_class=None,
                 g=9.81,
                 drag_coeff=0.0,
                 constant_disturbance=None,
                 initial_pos=None,
                 initial_vel=None):
        if drone_class is None:
            self.Drone = Drone()
        else:
            self.Drone = drone_class()
        self.m = self.Drone.m
        self.g = g
        self.drag_coeff = drag_coeff

        # Disturbance
        if constant_disturbance is None:
            self.constant_disturbance = np.array([0.0, 0.0, 0.0])
        else:
            self.constant_disturbance = np.array(constant_disturbance)

        # State
        if initial_pos is None:
            self.pos = np.array([0.0, 0.0, 0.0])
        else:
            self.pos = np.array(initial_pos)

        if initial_vel is None:
            self.vel = np.array([0.0, 0.0, 0.0])
        else:
            self.vel = np.array(initial_vel)

    def reset(self, pos=None, vel=None):
        """Reset state to initial conditions"""
        if pos is not None:
            self.pos = np.array(pos)
        else:
            self.pos = np.array([0.0, 0.0, 0.0])

        if vel is not None:
            self.vel = np.array(vel)
        else:
            self.vel = np.array([0.0, 0.0, 0.0])

    def step(self, F_control, dt):
        """
        Simulate one time step with applied control force.

        Args:
            F_control: Control force [Fx, Fy, Fz] in N (world frame)
            dt: Time step in seconds

        Returns:
            state: Current state [x, y, z, vx, vy, vz]
        """
        # Gravity force
        F_gravity = np.array([0.0, 0.0, -self.m * self.g])

        # Drag force (proportional to velocity)
        F_drag = -self.drag_coeff * self.vel

        # Total force
        F_total = F_control + F_gravity + F_drag + self.constant_disturbance

        # Acceleration
        acc = F_total / self.m

        # Integrate using Euler method
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        return self.get_state()

    def get_state(self):
        """Return current state [x, y, z, vx, vy, vz]"""
        return np.concatenate([self.pos, self.vel])

    def get_pos(self):
        """Return current position [x, y, z]"""
        return self.pos.copy()

    def get_vel(self):
        """Return current velocity [vx, vy, vz]"""
        return self.vel.copy()


class SimpleQuadcopterWithUnknownParams(SimpleQuadcopter):
    """
    Quadcopter with unknown parameters for testing adaptive control.
    Actual mass and disturbances differ from nominal values.
    """

    def __init__(self,
                 m_true=2.5,
                 m_nominal=1.9,
                 g=9.81,
                 drag_coeff=0.1,
                 constant_disturbance=None,
                 initial_pos=None,
                 initial_vel=None):
        """
        Args:
            m_true: True mass (unknown to controller)
            m_nominal: Nominal mass (known to controller)
            Other args same as SimpleQuadcopter
        """
        # Initialize with true mass
        super().__init__(m=m_true,
                        g=g,
                        drag_coeff=drag_coeff,
                        constant_disturbance=constant_disturbance,
                        initial_pos=initial_pos,
                        initial_vel=initial_vel)

        self.m_nominal = m_nominal
        self.m_true = m_true

    def get_nominal_mass(self):
        """Return nominal mass (what controller thinks mass is)"""
        return self.m_nominal

    def get_true_mass(self):
        """Return true mass (actual system mass)"""
        return self.m_true


if __name__ == "__main__":
    # Test the environment
    print("Testing Simple Quadcopter Environment\n")

    # Test 1: Hover with perfect knowledge
    print("Test 1: Hover with perfect mass knowledge")
    quad = SimpleQuadcopter(m=2.0, g=9.81)
    quad.reset(pos=[0, 0, 1])

    dt = 0.01
    F_hover = np.array([0.0, 0.0, quad.m * quad.g])

    for i in range(100):
        state = quad.step(F_hover, dt)

    print(f"  After 1s of hover control:")
    print(f"    Position: {quad.pos}")
    print(f"    Velocity: {quad.vel}")
    print(f"    (Should stay near [0, 0, 1] with zero velocity)\n")

    # Test 2: Hover with unknown mass
    print("Test 2: Hover with unknown mass (true=2.5kg, think=2.0kg)")
    quad2 = SimpleQuadcopterWithUnknownParams(m_true=2.5, m_nominal=2.0, g=9.81)
    quad2.reset(pos=[0, 0, 1])

    F_hover_wrong = np.array([0.0, 0.0, quad2.m_nominal * quad2.g])  # Using wrong mass

    for i in range(100):
        state = quad2.step(F_hover_wrong, dt)

    print(f"  After 1s of hover control (with wrong mass):")
    print(f"    Position: {quad2.pos}")
    print(f"    Velocity: {quad2.vel}")
    print(f"    (Should drift downward due to insufficient thrust)\n")

    # Test 3: Constant disturbance
    print("Test 3: Unknown constant disturbance")
    quad3 = SimpleQuadcopter(m=2.0, g=9.81, constant_disturbance=[2.0, 0.0, 0.0])
    quad3.reset(pos=[0, 0, 1])

    F_hover = np.array([0.0, 0.0, quad3.m * quad3.g])

    for i in range(100):
        state = quad3.step(F_hover, dt)

    print(f"  After 1s with constant x-disturbance (2N):")
    print(f"    Position: {quad3.pos}")
    print(f"    Velocity: {quad3.vel}")
    print(f"    (Should drift in +x direction)\n")
