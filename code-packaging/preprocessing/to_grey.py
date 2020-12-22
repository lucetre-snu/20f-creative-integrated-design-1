import os
from PIL import Image
import numpy as np
from shutil import copyfile

def convert_jpg_to_grey(image_folder_path, output_dir, option):

    src_dir = image_folder_path + '/' + option + '/'
    img_list = os.listdir(src_dir)
    img_list = [img_file_name for img_file_name in img_list if img_file_name.endswith('.jpg')]

    for img_file_name in img_list:
        img = Image.open(src_dir + img_file_name).convert('L')
        saved_name = output_dir +'/' + option + '/' + img_file_name
        img.save(saved_name)

def copy_npy(image_folder_path, output_dir, option):

    src_dir = image_folder_path + '/' + option + '/'
    label_list = os.listdir(src_dir)
    label_list = [label_file_name for label_file_name in label_list if label_file_name.endswith('.npy')]

    for label_file_name in label_list:
        copyfile(src_dir + label_file_name, output_dir + '/' + option + '/' + label_file_name)

def main(image_folder_path,output_dir):

    dir_list = os.listdir(image_folder_path)
    for dir_name in dir_list:
        convert_jpg_to_grey(image_folder_path, output_dir, dir_name)
        copy_npy(image_folder_path,output_dir,dir_name)



if __name__ == '__main__':
    main('../new_data','../greyscale_data')
    
