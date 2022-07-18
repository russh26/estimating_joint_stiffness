import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz

import os

os.chdir("C:\Repositories\estimating_joint_stiffness\data_csv")


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def file_plot(name_file, format):
    data = np.genfromtxt(name_file, delimiter=",", skip_header=True)

    time = []
    for index in range(len(data[:, 0])):
        if format == "mot":
            time.append(index * (1 / 480))
        else:
            time.append(index * (1 / 2000))
    time = np.array(time)
    if format == "mot":
        data = data[:, 3]
    else:
        data = np.vstack((data[:, 4], data[:, 7])).T

        data = np.abs(data)

        data = butter_lowpass_filter(data, 20, 2000, order=5)
    plt.xlabel("time")
    plt.ylabel("radians")
    plt.plot(time, data)
    if format == "mot":

        plt.legend(["elbow_flexion"])
        plt.savefig("pictures/euler_mot.png")

    else:

        plt.legend(["TriLong", "BicepLong"])
        plt.savefig("pictures/emg_sto.png")

    plt.show()


file_plot("new_emg.csv", "sto")
# file_plot('new_euler.csv', 'mot')


def sto_plot(name_file, format):
    with open(name_file, "r") as file:
        file = file.read().split("\n")[7:-1]

        time = []
        trilat = []
        biclong = []
        elbow_flex = []
        for line in file:
            line = line.split("\t")
            time.append(float(line[0].strip()))
            trilat.append(float(line[1].strip()))
            biclong.append(float(line[2].strip()))
            elbow_flex.append(float(line[3].strip()))
        time = np.array(time)
        elbow_flex = np.array(elbow_flex)
        muscles = np.vstack((np.array(trilat), np.array(biclong))).T

        plt.xlabel("time")
        plt.ylabel("radians")
        if format == "mot":
            plt.plot(time, elbow_flex)
            plt.legend(["elbow_flexion_reserve"])
            plt.savefig("pictures/elbow_flexion_sto.png")
        else:
            plt.plot(time, muscles)
            plt.legend(["TRIlat", "BIClong"])
            plt.savefig("pictures/muscles_sto.png")
        plt.show()


# sto_plot('model_controls.sto', 'mot')
# sto_plot('model_controls.sto', 'sto')

