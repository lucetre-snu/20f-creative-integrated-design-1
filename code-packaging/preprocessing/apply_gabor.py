import os

import numpy as np
import cv2
import math

def main(image_folder_path, save_folder_path):
    print("Processing images ...")
    img_list = os.listdir(image_folder_path)
    img_list = [img_file_name for img_file_name in img_list if img_file_name.endswith(".jpg")]
    img_list = sorted(img_list)

    # (Size ksize, double sigma, double theta, double lambd, double gamma, double psi=CV_PI *0.5, int ktype=CV_64F)
    kernel1 = cv2.getGaborKernel((21,21), 5, np.pi/4, 15, 1, 0, cv2.CV_32F)
    kernel1 /= math.sqrt((kernel1 * kernel1).sum())

    for img_file_name in img_list:
        print("Processing " + img_file_name + " ...")
        img = cv2.imread(image_folder_path + img_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_npy = np.array(img)
        filtered = cv2.filter2D(img_npy, -1, kernel1)
        cv2.imwrite(save_folder_path + "image/" + img_file_name, filtered)
        print(filtered)


if __name__ == "__main__":
    main("../cropped_data/image/", "../gabor/")
