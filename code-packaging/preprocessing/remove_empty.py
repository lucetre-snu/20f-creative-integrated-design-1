import os

import numpy as np
from PIL import Image


def main(numpy_folder_path):
    f = open(numpy_folder_path + "empty.txt", 'r')
    i = 0
    while True:
        line = f.readline()
        line = line.rstrip('\n')
        if not line:
            break
        print(line)
        os.remove(numpy_folder_path + line + ".msk")
        os.remove(numpy_folder_path + line + ".jpg")
        i += 1
    print("total " + str(i) + " files are removed")


if __name__ == "__main__":
    main("../FaceScape_noempty/",)
