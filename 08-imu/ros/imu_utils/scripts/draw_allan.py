
# command-line arguments:
import argparse
# data handling:
import pandas as pd
import numpy as np
# visualization:
import matplotlib.pyplot as plt

IMU_NAME = "VIOSim"

def load_imu_data(imu_name):
    # load dataset from output files:
    data = {}
    for attribute in ("gyr_t","gyr_x","gyr_y","gyr_z","sim_gyr_x","sim_gyr_y","sim_gyr_z"
    ):
        with open("../data/data_{}_{}.txt".format(imu_name, attribute), "rt") as f:
            values = [float(a) for a in f.readlines()]
        data[attribute] = values
    
    # format as Pandas dataframe:
    data = pd.DataFrame.from_dict(data)

    return data

def draw_allan_plot(imu, df_imu):
    # plot data:
    plt.loglog(df_imu["gyr_t"], df_imu["gyr_x"], "ro", label = "Gyr X", markersize = 1)
    plt.loglog(df_imu["gyr_t"], df_imu["gyr_y"], "go", label = "Gyr Y", markersize = 1)
    plt.loglog(df_imu["gyr_t"], df_imu["gyr_z"], "bo", label = "Gyr Z", markersize = 1)
    plt.loglog(df_imu["gyr_t"], df_imu["sim_gyr_x"], "r-", label = "Sim Gyr X", markersize = 1)
    plt.loglog(df_imu["gyr_t"], df_imu["sim_gyr_y"], "g-", label = "Sim Gyr Y", markersize = 1)
    plt.loglog(df_imu["gyr_t"], df_imu["sim_gyr_z"], "b-", label = "Sim Gyr Z", markersize = 1)
    # title:
    plt.title("Allan Variance Curve: {}".format(imu))
    # axis labels:
    plt.xlabel("Time:Sec")
    plt.ylabel("Sigma:Deg/H")
    # legend:
    plt.legend(loc="upper left")
    # show grid:
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # IMU selection:
    parser = argparse.ArgumentParser(description='Draw Allan variance curve.')
    parser.add_argument('--imu', dest='imu',
                        default=IMU_NAME,
                        help='Targe IMU to be analyzed.')
    args = parser.parse_args()

    # load IMU data:
    df_imu = load_imu_data(args.imu)
    # draw Allan variance curve:
    draw_allan_plot(args.imu, df_imu)
