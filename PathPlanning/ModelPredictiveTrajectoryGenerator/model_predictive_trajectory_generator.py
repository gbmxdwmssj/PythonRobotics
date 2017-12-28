"""
Model trajectory generator

author: Atsushi Sakai

the design of p can be improved, only two steer, v is connstant, no acceleration and deacceleration

Jacobian: the calculation of Jacobian may be not right? (if steer limit is small, can't optimize!? or cost threshold too small?),
Jacobian: up and down on the starting of optimization is also a problem
Jacobian: what is the optimization's theory?
Jacobian: if the initial path is too long, can't optimize!?
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math
import motion_model

## get directory of matplotrecorder
github_root = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.append(github_root)

from matplotrecorder import matplotrecorder

# optimization parameter
max_iter = 150
h = np.matrix([0.5, 0.02, 0.02]).T  # parameter sampling distance
cost_th = 0.1

matplotrecorder.DO_NOTHING = True
show_graph = True

def limitP(p, cfg):
    p[1, 0] = np.clip(p[1, 0], cfg.min_steer, cfg.max_steer)
    p[2, 0] = np.clip(p[2, 0], cfg.min_steer, cfg.max_steer)

    return p

def limitDp(dp, cfg, p):
    dp[1, 0] = np.clip(dp[1, 0], cfg.min_steer - p[1, 0], cfg.max_steer - p[1, 0])
    dp[2, 0] = np.clip(dp[2, 0], cfg.min_steer - p[2, 0], cfg.max_steer - p[2, 0])

    return dp

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    u"""
    Plot arrow
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)
    plt.plot(0, 0)


def calc_diff(target, x, y, yaw):
    d = np.array([target.x - x[-1],
                  target.y - y[-1],
                  motion_model.pi_2_pi(target.yaw - yaw[-1])])

    return d


def calc_J(target, p, h, k0):
    ## change of x (R^3) is the function of p(R^3)

    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0] + h[0, 0], p[1, 0], p[2, 0], k0)
    dp = calc_diff(target, [xp], [yp], [yawp]) # Eq. (13) in paper
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0] - h[0, 0], p[1, 0], p[2, 0], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d1 = np.matrix((dp - dn) / (2.0 * h[1, 0])).T # Eq. (16) in paper

    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0], p[1, 0] + h[1, 0], p[2, 0], k0)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0], p[1, 0] - h[1, 0], p[2, 0], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d2 = np.matrix((dp - dn) / (2.0 * h[2, 0])).T

    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] + h[2, 0], k0)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] - h[2, 0], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d3 = np.matrix((dp - dn) / (2.0 * h[2, 0])).T

    J = np.hstack((d1, d2, d3))

    return J


def selection_learning_param(dp, p, k0, target):

    mincost = float("inf")
    mina = 0.6
    maxa = 1.4 # for np.arange, can not reach max
    da = 0.2

    for a in np.arange(mina, maxa, da):
        tp = p[:, :] + a * dp
        xc, yc, yawc = motion_model.generate_last_state(
            tp[0], tp[1], tp[2], k0)
        dc = np.matrix(calc_diff(target, [xc], [yc], [yawc])).T
        cost = np.linalg.norm(dc)

        if cost <= mincost and a != 0.0:
            mina = a
            mincost = cost

    #  print(mincost, mina)
    #  input()

    return mina


def show_trajectory(target, xc, yc):

    plt.clf()
    plot_arrow(target.x, target.y, target.yaw)
    plt.plot(xc, yc, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.1)
    matplotrecorder.save_frame()


def optimize_trajectory(target, k0, p):
    mcfg = motion_model.ModelConfig()
    for i in range(max_iter):
        xc, yc, yawc = motion_model.generate_trajectory(p[0], p[1], p[2], k0)
        dc = np.matrix(calc_diff(target, xc, yc, yawc)).T
        #  print(dc.T)

        cost = np.linalg.norm(dc)
        if cost <= cost_th:
            ## optimize success
            print("path is ok cost is:" + str(cost))
            break

        J = calc_J(target, p, h, k0)
        try:
            dp = - np.linalg.inv(J) * dc # Eq. (14) in paper
            dp = limitDp(dp, mcfg, p)
        except np.linalg.linalg.LinAlgError:
            ## optimize fail
            warnings.warn("cannot calc path LinAlgError")
            xc, yc, yawc, p = None, None, None, None
            break
        alpha = selection_learning_param(dp, p, k0, target) # choose learning rate

        p += alpha * np.array(dp)
        p = limitP(p, mcfg)
        #  print(p.T)

        if show_graph:
            show_trajectory(target, xc, yc)
    else:
        ## optimize fail
        ## if no break, enter here
        xc, yc, yawc, p = None, None, None, None
        warnings.warn("cannot calc path")

    return xc, yc, yawc, p


def test_optimize_trajectory():
    mcfg = motion_model.ModelConfig()

    #  target = motion_model.State(x=5.0, y=2.0, yaw=math.radians(00.0))
    target = motion_model.State(x=5.0, y=2.0, yaw=math.radians(0.0))
    k0 = 0.0

    init_p_len = math.sqrt(target.x**2 + target.y**2)
    init_p = np.matrix([init_p_len, 0.0, 0.0]).T
    init_p = limitP(init_p, mcfg)

    x, y, yaw, p = optimize_trajectory(target, k0, init_p)
    p = limitP(p, mcfg)

    show_trajectory(target, x, y)
    matplotrecorder.save_movie("animation.mp4", 0.1)
    #  plt.plot(x, y, "-r")
    plot_arrow(target.x, target.y, target.yaw)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def test_trajectory_generate():
    s = 5.0  # [m]
    k0 = 0.0
    km = math.radians(30.0)
    kf = math.radians(-30.0)

    # plt.plot(xk, yk, "xr")
    # plt.plot(t, kp)
    # plt.show()

    x, y, _ = motion_model.generate_trajectory(s, km, kf, k0)

    plt.plot(x, y, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def main():
    print(__file__ + " start!!")
    # test_trajectory_generate()
    test_optimize_trajectory()


if __name__ == '__main__':
    main()
