"""
Lookup Table generation for model predictive trajectory generator

author: Atsushi Sakai

The initial state is assumed to be (0, 0, 0) in this program. Other initial states can be rotated and translated to (0, 0, 0).
"""
from matplotlib import pyplot as plt
import numpy as np
import math
import model_predictive_trajectory_generator as planner
import motion_model
import pandas as pd


def calc_states_list():
    maxyaw = math.radians(-30.0)

    x = np.arange(1.0, 30.0, 5.0)
    y = np.arange(0.0, 20.0, 2.0)
    yaw = np.arange(-maxyaw, maxyaw, maxyaw)

    states = []
    for iyaw in yaw:
        for iy in y:
            for ix in x:
                states.append([ix, iy, iyaw])
    # print(len(states))

    return states


def search_nearest_one_from_lookuptable(tx, ty, tyaw, lookuptable):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookuptable):

        dx = tx - table[0]
        dy = ty - table[1]
        dyaw = tyaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2) # Euclidean norm
        if d <= mind:
            minid = i
            mind = d

    # print(minid)

    return lookuptable[minid]


def save_lookup_table(fname, table):
    mt = np.array(table)
    # print(mt)
    # save csv
    df = pd.DataFrame()
    df["x"] = mt[:, 0]
    df["y"] = mt[:, 1]
    df["yaw"] = mt[:, 2]
    df["s"] = mt[:, 3]
    df["km"] = mt[:, 4]
    df["kf"] = mt[:, 5]
    # df.to_csv(fname, index=None)
    df.to_csv(fname)

    print("lookup table file is saved as " + fname)


def generate_lookup_table():
    states = calc_states_list() # calculate target states
    k0 = 0.0

    # target_x, target_y, target_yaw -> s, km, kf
    lookuptable = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

    print("Lookup Table is being generated......")
    
    for idx, state in enumerate(states):
        bestp = search_nearest_one_from_lookuptable(
            state[0], state[1], state[2], lookuptable)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        # init_p = np.matrix(
        #     [math.sqrt(state[0] ** 2 + state[1] ** 2), bestp[4], bestp[5]]).T
        init_p = np.matrix(
            [bestp[3], bestp[4], bestp[5]]).T

        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            print("find good path")
            lookuptable.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])

        ## print percent of progress
        percent = 100 * (idx + 1.0) / len(states)
        print("Complete %{}...".format(percent))

    print("finish lookup table generation")

    save_lookup_table("lookuptable.csv", lookuptable)

    for table in lookuptable:
        xc, yc, yawc = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        plt.plot(xc, yc, "-r")
        xc, yc, yawc = motion_model.generate_trajectory(
            table[3], -table[4], -table[5], k0) # symmetrical, this is why target_yaw inlcude only -30 degree and exclude +30 degree
                                                # similar for target_y, note that not for target_x
                                                # also note that here change the sign of steering make the sign of target_yaw
                                                # and target_y change at the same time
        plt.plot(xc, yc, "-r")

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


def main():
    generate_lookup_table()


if __name__ == '__main__':
    main()
