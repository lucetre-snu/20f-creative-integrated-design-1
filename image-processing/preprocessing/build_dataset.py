import argparse
import random
import os
from PIL import Image
import numpy as np
from shutil import copyfile

def save(image_folder_path,id_list,output_dir):
    
    # .jpg directory
    texture_dir = image_folder_path + '/texture-jpg'
    texture_filenames = os.listdir(texture_dir)
    # .npy directory
    label_dir = image_folder_path + '/label-npy'
    label_filenames = os.listdir(label_dir)

    for id in id_list:
        for filename in texture_filenames:
            if(filename.split("_")[0] == id):
                #print(texture_dir + '/' + filename)
                copyfile(texture_dir + '/' + filename,output_dir + '/' + filename)
                #image = Image.open(filename)
                #image.save(output_dir)
        for filename in label_filenames:
            if(filename.split("_")[0] == id):
                copyfile(label_dir + '/' + filename, output_dir + '/' + filename)
                #np.save(output_dir,label_dir + '/' + filename)
                #image.save(output_dir)


def main(image_folder_path, output_folder_path):

    # Get distinct people and split them into three groups
    id_list = list()
    cnt = 0
    for filename in os.listdir(image_folder_path + '/texture-jpg'):

        if(filename.endswith(".jpg")):
            id_list.append(filename.split("_")[0]);
            cnt += 1

    distinct_id_list = list(set(id_list))
    distinct_id_list.sort()
    print(distinct_id_list)

    random.seed(230)
    random.shuffle(distinct_id_list)

    split_1 = int(0.8*len(distinct_id_list))
    split_2 = int(0.9*len(distinct_id_list))

    train_ids = distinct_id_list[:split_1]
    print(train_ids)
    val_ids = distinct_id_list[split_1:split_2]
    print(val_ids)
    test_ids = distinct_id_list[split_2:]
    print(test_ids)

    # Define the output data directories
    data_dir = os.path.join(output_folder_path,'dataset')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    else:
        print("Directory already exists")
        #return

    train_data_dir = os.path.join(data_dir,'train')
    val_data_dir = os.path.join(data_dir,'val')
    test_data_dir = os.path.join(data_dir,'test')

    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    else:
        print("Directory already exists")
 
    if not os.path.exists(val_data_dir):
        os.mkdir(val_data_dir)
    else:
        print("Directory already exists")  
    
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)
    else:
        print("Directory already exists")

    # save files
    save(image_folder_path, train_ids, train_data_dir)
    save(image_folder_path, val_ids, val_data_dir)
    save(image_folder_path, test_ids, test_data_dir)
        

if __name__ == "__main__":
    main("../DATA/CROPPED","../../wrinkle-detection")
