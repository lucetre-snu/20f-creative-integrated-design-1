import os

import numpy as np
from PIL import Image


def main(numpy_folder_path):

    label_list = os.listdir(numpy_folder_path)
    label_list = [label_file_name for label_file_name in label_list if label_file_name.endswith(".npy")]
    label_list = sorted(label_list)
    for label_file_name in label_list:
        label = np.load(numpy_folder_path + label_file_name)

        filename_without_ext = label_file_name.split(".")[0]
        if np.sum(label) < 10:
            print(filename_without_ext)


if __name__ == "__main__":
    main("../label_data/numpy/",)
