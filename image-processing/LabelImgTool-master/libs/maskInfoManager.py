import struct
import codecs
import os
import re
import sys
import time

from PyQt4.QtGui import *
from PyQt4.QtCore import *

class MaskInfoManager():
    
    def __init__(self, *args, **kwargs):
        pass
    
    def saveMaskFile(self, imgFileName, imgRes, numMaskLayer, imgBuf) :
        fileDir, imgFile = os.path.split(imgFileName)
        maskFileName = imgFile.replace(imgFile.split(".")[-1], "msk")

        print "imageSize : " + str(imgRes[1]) + ", " + str(imgRes[0])
        totalSize = imgRes[0] * imgRes[1]
        bufSize = totalSize/32
        if totalSize%32 :
            bufSize += 1
        maskBuf = [0] * bufSize * numMaskLayer
        print "buffer size : " + str(bufSize)

        byteNum = 0
        for i in range(numMaskLayer) :
            bitPos = 1
            byteInfo = 0
            for y in range(imgRes[0]) :
                for x in range(imgRes[1]) :                    
                    alphaVal = QColor.fromRgba(imgBuf[i].pixel(x, y)).alpha()
                    if alphaVal :
                        byteInfo |= bitPos
                    if bitPos == 0x80000000 :
                        maskBuf[byteNum] = byteInfo
                        bitPos = 1
                        byteInfo = 0
                        byteNum+=1
                    else :
                        bitPos = bitPos << 1
                    
            if (bitPos!=1) :
                maskBuf[byteNum] = byteInfo
                byteNum+=1

        if not os.path.isdir(fileDir) :
            os.mkdir(fileDir)
        
        f = open(fileDir + '/' + maskFileName,'wb')
        f.write(struct.pack('iii', imgRes[1], imgRes[0], numMaskLayer))
        f.write(struct.pack('%dI'%(bufSize*numMaskLayer), *maskBuf))
        f.close()

    def loadMaskFile(self, imgFileName, imgRes, maskColor, eraseColor) :
        fileDir, imgFile = os.path.split(imgFileName)
        maskFileName = imgFile.replace(imgFile.split(".")[-1], "msk")

        print "imageSize : " + str(imgRes[1]) + ", " + str(imgRes[0])
        totalSize = imgRes[0] * imgRes[1]
        bufSize = totalSize/32
        if totalSize%32 :
            bufSize += 1
        print "buffer size : " + str(bufSize)

        if not os.path.isfile(fileDir + '/' + maskFileName) :
            print "Error : There's no file '%s'."%(fileDir + '/' + maskFileName)
            return -1, None

        f = open(fileDir + '/' + maskFileName,'rb')
        fData = f.read()
        fResX, fResY, numMaskLayer = struct.unpack('iii', fData[0:12])
        if imgRes[1] != fResX :
            print "Error : There's information mismatch to image width(%d)."%fResX
            return -1, None
        if imgRes[0] != fResY :
            print "Error : There's information mismatch to image height(%d)."%fResY
            return -1, None

        retImages = []
        for i in range(numMaskLayer) :
            retImages.append(QImage(QSize(fResX, fResY), QImage.Format_ARGB32))
            retImages[i].fill(eraseColor)

        index = 3
        for i in range(numMaskLayer) :
            bitPos = 1
            unitInfo = int(struct.unpack('I', fData[index*4:(index+1)*4])[0])
            for y in range(fResY) :
                for x in range(fResX) :                    
                    if (unitInfo & bitPos) :
                        retImages[i].setPixel(x, y, maskColor.rgba())
                        
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

        return numMaskLayer, retImages
    




