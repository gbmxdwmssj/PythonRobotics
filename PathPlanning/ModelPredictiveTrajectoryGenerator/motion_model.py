import math
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# motion parameter
L = 1.0  # wheel base
ds = 0.1  # course distanse
v = 10.0 / 3.6  # velocity [m/s]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class ModelConfig:

    def __init__(self):
        self.min_steer = math.radians(-40.0)
        self.max_steer = math.radians(40.0)


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def update(state, v, delta, dt, L):

    ## Ackermann model
    ## limit amplitude
    ## TODO: actual path length isn't equal to path length (s or p[0]) in parameterized control sequence (p)?

    model_cfg = ModelConfig()
    delta = np.clip(delta, model_cfg.min_steer, model_cfg.max_steer) # steering min and max [rad]

    state.v = v
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.yaw = pi_2_pi(state.yaw)

    return state


def generate_trajectory(s, km, kf, k0):

    n = s / ds
    time = s / v  # [s]
    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    kp = scipy.interpolate.spline(tk, kk, t, order=2)
    dt = float(time / n) # discretized by equal distance "ds"

    # plt.plot(t, kp)
    # plt.show()

    state = State()
    x, y, yaw = [state.x], [state.y], [state.yaw]

    for ikp in kp:
        state = update(state, v, ikp, dt, L)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)

    return x, y, yaw


def generate_last_state(s, km, kf, k0):

    n = s / ds
    time = s / v  # [s]
    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    kp = scipy.interpolate.spline(tk, kk, t, order=2)
    dt = time / n

    #  plt.plot(t, kp)
    #  plt.show()

    state = State()

    [update(state, v, ikp, dt, L) for ikp in kp] # for class, no return is ok, this has been tested.
    return state.x, state.y, state.yaw
