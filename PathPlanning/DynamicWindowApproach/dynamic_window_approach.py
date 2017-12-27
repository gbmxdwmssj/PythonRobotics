"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi)

local minimum is the natural disadvantage of this method, global planning can relieve it partly

local minimum is also sensitive to the weights in cost function, we can adjust the weights online by some method, such as RL.

0 is 0, inf is inf, don't use small or big value to replace, use boolean value or floa("Inf")

"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

## get directory of matplotrecorder
github_root = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.append(github_root)

from matplotrecorder import matplotrecorder
matplotrecorder.donothing = False


class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]

        self.max_accel = 0.2  # [m/ss]
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]

        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]

        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]

        self.to_goal_cost_gain = 0.2
        self.speed_cost_gain = 0.45
        self.obs_cost_gain = 7.0

        self.robot_radius = 1.0  # [m]
        self.goal_area = 0.3 # [m]


def motion(x, u, dt):
    # motion model

    ## TODO: real robot's state is sensed, not set by your control command

    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]
    # print(Vs, Vd)

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    # print(dw)

    return dw

def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    #  print(len(traj))
    return traj


def calc_final_input(x, u, dw, config, goal, ob):

    xinit = x[:]
    min_cost = float("Inf")
    min_u = u
    min_u[0] = 0.0 # real collision make the robot stop
                   # not necessary to keep a non-zero yawrate to escape from collision
                   # it will automatically change to a non-zero value
                   # this is the so called rotate away mode in original paper
                   # TODO: note that the inside of the current robot state should be forced to be collision free
    best_traj = np.array([x])

    # evalucate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(xinit, v, y, config)

            ## TODO: normalization in cost
            ## calc cost
            ## dynamic window only consider one dt
            ## but evaluation should be conducted on the whole trajectory
            ## TODO: consider orientation relative to goal in goal cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, goal, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3]) # TODO: neg. speed is penalized here, is this right?
            ## TODO: do what the original paper says in obstacle cost
            ob_cost = config.obs_cost_gain * calc_obstacle_cost(traj, ob, config)
            #  print(ob_cost)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                min_u = [v, y]
                best_traj = traj

    #  print(min_u)
    #  input()

    return min_u, best_traj


def calc_obstacle_cost(traj, ob, config):
    # calc obstacle cost inf: collistion, 0:free

    skip_n = 2
    minr = float("inf")

    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in range(len(ob[:, 0])):
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2) - config.robot_radius
            if r <= 0 and ii != 0: # ignore the current state
                return float("Inf")  # collisiton

            if minr >= r:
                minr = r

    ## TODO: we should consider the distance between the margins of robot and obstacle
    return 1.0 / minr  # OK


def calc_to_goal_cost(traj, goal, config):
    # calc to goal cost. It is 2D norm.

    dy = goal[0] - traj[-1, 0]
    dx = goal[1] - traj[-1, 1]
    goal_dis = math.sqrt(dx**2 + dy**2)
    cost = goal_dis

    return cost


def dwa_control(x, u, config, goal, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u, traj = calc_final_input(x, u, dw, config, goal, ob)

    return u, traj


def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def main():
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 2.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([0, 20])
    # obstacles [x(m) y(m), ....]
    ob = np.matrix([[-4.0, 5.0],
                    [-4.0, 6.0],
                    [-4.0, 7.0],
                    [-4.0, 8.0],
                    [-4.0, 9.0],
                    [-4.0, 10.0],
                    [-3.0, 10.0],
                    [-2.0, 10.0],
                    [-1.0, 10.0],
                    [0.0, 10.0],
                    [1.0, 10.0],
                    [2.0, 10.0],
                    [3.0, 10.0],
                    [4.0, 10.0],
                    [5.0, 10.0],
                    [6.0, 10.0],
                    [7.0, 10.0],
                    [8.0, 10.0],
                    [9.0, 10.0]
                    ])

    u = np.array([0.0, 0.0])
    config = Config()
    traj = np.array(x)

    for i in range(1000):
        print("################# STEP {} #################".format(i))

        # if i == 330:
        #     config.to_goal_cost_gain = 0.0
        #     config.speed_cost_gain = 0.0
        #     config.obs_cost_gain = 999
        #     print("################# escape ################# ")
        
        plt.cla()
        u, ltraj = dwa_control(x, u, config, goal, ob)

        x = motion(x, u, config.dt)
        traj = np.vstack((traj, x))  # store state history

        plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
        plt.plot(x[0], x[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        plt.plot(ob[:, 0], ob[:, 1], "ok")
        plot_arrow(x[0], x[1], x[2])

        ## print v and yawrate
        print("v = {:.5f}m/s\tyawrate = {:.5f}rad/s".format(x[3], x[4]))

        ## darw circle ((x[0], x[1]), config.robot_radius)
        angles_circle = [kk * math.pi / 180.0 for kk in range(0, 360)]
        cir_x = config.robot_radius * np.cos(angles_circle) + x[0]
        cir_y = config.robot_radius * np.sin(angles_circle) + x[1]
        plt.plot(cir_x, cir_y, '-r')

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        matplotrecorder.save_frame()

        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.goal_area:
            print("Goal!!")
            break

    print("Done")
    plt.plot(traj[:, 0], traj[:, 1], "-r")
    matplotrecorder.save_frame()
    matplotrecorder.save_movie("animation.gif", config.dt, monitor = False)
    plt.show()


if __name__ == '__main__':
    main()
