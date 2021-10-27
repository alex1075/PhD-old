import cv2
import glob, os
from PIL import Image
import numpy as np

#Grabs biggest dimension and scales the photo so that max dim is now 1280
def resizeTo(image, newhigh=1280, newwid=1280, inter=cv2.INTER_AREA):
    (height, width) = image.shape[:2]
    if height>width:
        newheight = newhigh
        heightratio = height/newheight
        newwidth = int(width/heightratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    elif width>height:
        newwidth = newwid
        widthratio = width/newwidth
        newheight = int(height/widthratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    else: 
        print('Error')


def normaliseImg(self, img):
    if np.max(img)==np.min(img):
        img_norm = (img - np.min(img)) / (np.max(img+1) - np.min(img-1))
    else:
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (img_norm * 2) - 1

def softNormaliseImg(self, img):
    pmin = np.percentile(img, 1)
    img_norm = ((img - pmin)/(np.percentile(img, 99) - pmin))
    return (img+1)/2

def normaliseImgBack(self, img):    
    return (img+1)/2

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def rgb2gray(img):
    img = cv2.imread(img,0)
    return img