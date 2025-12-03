import numpy as np

def triangular_trajectory(t):
    period = 100.0

    t_mod = t % period

    if t_mod < 50:
        x_d = 0.0
    elif t_mod < 100:
        x_d = 10.0
    elif t_mod < 150:
        x_d = 10.0
    elif t_mod < 200:
        x_d = 10.0
    elif t_mod < 250:
        x_d = 10.0
    else:
        x_d = 0.0

    if t_mod < 50:
        y_d = 0.0
    elif t_mod < 100:
        y_d = (t_mod - 50) / 50 * 6.0
    elif t_mod < 150:
        y_d = 6.0
    elif t_mod < 200:
        y_d = 6.0 - (t_mod - 150) / 50 * 6.0
    elif t_mod < 250:
        y_d = 0.0
    else:
        y_d = 0.0

    if t_mod < 50:
        z_d = t_mod / 50 * 5.0
    elif t_mod < 100:
        z_d = 5.0
    elif t_mod < 150:
        z_d = 5.0 - (t_mod - 100) / 50 * 2.0
    elif t_mod < 200:
        z_d = 3.0
    elif t_mod < 250:
        z_d = 3.0 + (t_mod - 200) / 50 * 1.0
    else:
        z_d = 4.0

    psi_d = 0.0

    return np.array([x_d, y_d, z_d, psi_d])


def step_trajectory(t):
    if t < 2.0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    elif t < 5.0:
        return np.array([0.0, 0.0, 5.0, 0.0])
    elif t < 8.0:
        return np.array([10.0, 0.0, 5.0, 0.0])
    elif t < 11.0:
        return np.array([10.0, 6.0, 5.0, 0.0])
    elif t < 14.0:
        return np.array([10.0, 6.0, 3.0, 0.0])
    elif t < 17.0:
        return np.array([10.0, 0.0, 3.0, 0.0])
    else:
        return np.array([0.0, 0.0, 4.0, 0.0])


def hover_trajectory(t):
    return np.array([0.0, 0.0, 3.0, 0.0])


def circle_trajectory(t):
    radius = 2.0
    omega = 0.5
    height = 3.0

    x_d = radius * np.cos(omega * t)
    y_d = radius * np.sin(omega * t)
    z_d = height
    psi_d = 0.0

    return np.array([x_d, y_d, z_d, psi_d])
