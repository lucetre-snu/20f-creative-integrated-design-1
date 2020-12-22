import os

import numpy as np
from PIL import Image


def find_first_nonzero_index(file_name, numpy_array, direction):
    h, w = numpy_array.shape
    if direction == "right":
        for i in range(w):
            if np.sum(numpy_array[:, i]) != 0:
                return i
    elif direction == "left":
        for i in range(w):
            if np.sum(numpy_array[:, w-1-i]) != 0:
                return w-1-i
    elif direction == "down":
        for i in range(h):
            if np.sum(numpy_array[i, :]) != 0:
                return i
    elif direction == "up":
        for i in range(h):
            if np.sum(numpy_array[h-1-i, :]) != 0:
                return h-1-i
    else:
        assert False, "direction parameter must be one of up, down, right or left"

    print(file_name)
    assert False, "given numpy array is empty"


def main(data_folder):
    file_list = os.listdir(data_folder)
    file_list = [file for file in file_list if file.endswith(".npy")]

    x_min = 987654321
    x_max = -1
    y_min = 987654321
    y_max = -1

    left_average = 0.
    right_average = 0.
    down_average = 0.
    up_average = 0.
    index = 0.

    for file_name in file_list:
        npy_array = np.load(data_folder + file_name)
        index += 1
        right = find_first_nonzero_index(file_name, npy_array, "right")
        left = find_first_nonzero_index(file_name, npy_array, "left")
        down = find_first_nonzero_index(file_name, npy_array, "down")
        up = find_first_nonzero_index(file_name, npy_array, "up")
        print "Processing " + file_name + " \ right : " + str(right) + " ... \ left : " + str(left) + " \ down : " + str(down) + " \ up : " + str(up) + " \ "

        left_average = left_average * ((index -1) / index) + left / index
        right_average = right_average * ((index - 1) / index) + right / index
        up_average = up_average * ((index - 1) / index) + up / index
        down_average = down_average * ((index - 1) / index) + down / index

        if right < x_min:
            x_min = right
        if left > x_max:
            x_max = left
        if down < y_min:
            y_min = down
        if up > y_max:
            y_max = up
    
    print "average right : " + str(right_average) + ", left : " + str(left_average) + ", down : " + str(down_average) + ", up : " + str(up_average)
    print "x_min : " + str(x_min) + ", x_max : " + str(x_max) + ", y_min : " + str(y_min) + ", y_max : " + str(y_max)


if __name__ == "__main__":
    main("../label_data/numpy/")
