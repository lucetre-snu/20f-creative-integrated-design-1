import os

import numpy as np
from PIL import Image


def main(data_folder):
    file_list = os.listdir(data_folder)
    file_list = [file for file in file_list if file.endswith(".npy")]

    x_min = 987654321
    x_max = -1
    y_min = 987654321
    y_max = -1

    total_mean = 0.
    index = 0.

    for file_name in file_list:
        npy_array = np.load(data_folder + file_name)
        index += 1
        mean = np.mean(npy_array)
        total_mean += mean
        print "Processing " + file_name + " \ mean : " + str(mean)

    
    print "average mean : " + str(total_mean / index)


if __name__ == "__main__":
    main("../eye_data/train/")
