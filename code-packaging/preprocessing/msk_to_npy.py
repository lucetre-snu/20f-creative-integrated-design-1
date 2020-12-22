import struct
import codecs
import os
import re
import sys
import time

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy as np
from PIL import Image

def save_label_as_npy(source_folder_path, msk_file_name, save_folder_path):
        print "Processing " + msk_file_name + " ..."

        if not os.path.isfile(source_folder_path + '/' + msk_file_name) :
            print "Error : There's no file '%s'."%(fileDir + '/' + maskFileName)
            return -1, None

        f = open(source_folder_path + '/' + msk_file_name,'rb')
        fData = f.read()
        fResX, fResY, numMaskLayer = struct.unpack('iii', fData[0:12])

        totalSize = fResY * fResX
        bufSize = totalSize/32
        if totalSize%32 :
            bufSize += 1


        index = 3
        labelArray = np.zeros((fResY, fResX), dtype=int)
        for i in range(numMaskLayer) :
            bitPos = 1
            unitInfo = int(struct.unpack('I', fData[index*4:(index+1)*4])[0])
            for y in range(fResY) :
                for x in range(fResX) :                    
                    if (unitInfo & bitPos) :
                        labelArray[y, x] = 1
                        
                    if bitPos == 0x80000000 :
                        bitPos = 1
                        index+=1
                        if ((index)<bufSize*numMaskLayer+3) :
                            unitInfo = int(struct.unpack('I', fData[index*4:(index+1)*4])[0])
                    else :
                        bitPos = bitPos << 1
            
            if (bitPos!=1) :
                index+=1
        
        f.close()

        filename_without_ext = msk_file_name.split(".")[0]
        np.save(save_folder_path + "/numpy/" + filename_without_ext, labelArray)
        labelImg = Image.fromarray((labelArray * 255).astype(np.uint8))
        labelImg.save(save_folder_path + "/jpg/" + filename_without_ext + ".jpg")

def main(source_folder_path, save_folder_path):
    file_list = os.listdir(source_folder_path)
    file_list_msk = [file for file in file_list if file.endswith(".msk")]

    already_processed_files = set(os.listdir(save_folder_path + "/numpy/"))

    for file_name in file_list_msk:
        numpy_file_name = file_name.replace(file_name.split(".")[-1], "npy")
        if not numpy_file_name in already_processed_files:
            save_label_as_npy(source_folder_path, file_name, save_folder_path)
        else:
            print(file_name + " is already processed")

if __name__ == "__main__":
    main("../FaceScape_noempty/", "../label_data/")
