import os

import numpy as np
from PIL import Image


def main(image_folder_path, numpy_folder_path, save_folder_path, x_min, x_max, y_min, y_max):
    print("Processing images ...")
    img_list = os.listdir(image_folder_path)
    img_list = [img_file_name for img_file_name in img_list if img_file_name.endswith(".jpg")]
    for img_file_name in img_list:
        print("Processing " + img_file_name + " ...")
        img = Image.open(image_folder_path + img_file_name).convert('RGB')
        img_npy = np.array(img)
        cropped_img = img_npy[y_min:y_max, x_min:x_max]
        cropped_img = Image.fromarray(cropped_img.astype(np.uint8))
        cropped_img.save(save_folder_path + "/" + img_file_name)

    print("\n--------------------------------------------------------------\n")
    print("Processing images ...")

    label_list = os.listdir(numpy_folder_path)
    label_list = [label_file_name for label_file_name in label_list if label_file_name.endswith(".npy")]
    for label_file_name in label_list:
        print("Processing " + label_file_name + " ...")
        label = np.load(numpy_folder_path + label_file_name)
        cropped_label = label[y_min:y_max, x_min:x_max]

        #filename_without_ext = label_file_name.split(".")[0]
        #cropped_label_img = Image.fromarray((cropped_label * 255).astype(np.uint8))
        #cropped_label_img.save(save_folder_path + "/" + label_file_name)
        #cropped_label_img.save(save_folder_path + "/" + filename_without_ext + ".jpg")

        np.save(save_folder_path + label_file_name, cropped_label)


if __name__ == "__main__":
    main("../new_data/train/", "../new_data/train/", "./train/", 0, 1248, 850, 1330)
    main("../new_data/test/", "../new_data/test/", "./test/", 0, 1248, 850, 1330)
    main("../new_data/val/", "../new_data/val/", "./val/", 0, 1248, 850, 1330)
