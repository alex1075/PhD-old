import os, glob
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# Changes the images by cropping randomly through them
def getRandomCrop(self, img, crop_height, crop_width):
        max_x = np.shape(img)[1] - crop_width
        max_y = np.shape(img)[0] - crop_height
        x = np.random.randint(1, max_x-1)
        y = np.random.randint(1, max_y-1)
        crop = img[y:y+crop_height, x:x+crop_width]
        # Apply augmentation to sample        
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])  
        crop = seq.augment_image(crop)
        return np.expand_dims(crop, axis=-1)

# Change the contrast of an image (positive values to increase, negative values to decrease)
def imageContrastIncrease(self, img, intensity):
        ones = np.ones((img.shape[0], img.shape[0]))
        contrast_increase = ones*intensity
        increased = np.multiply(img,contrast_increase)
        if np.max(increased)>1:
            increased = np.clip(increased, -1, 1)
        return np.array(increased, dtype='float')

# Play with brightness
def imageBrightnessDecrease(self, img, intensity):
    img = np.array(img)
    print(np.max(img))
    print(img.shape)
    ones = np.ones((img.shape[0],img.shape[0]))
    brightness_decrease = ones*intensity
    decreased = img-brightness_decrease
    if np.min(decreased)<0:
        decreased = np.clip(decreased, -1, 1)
    return np.array(decreased, dtype='float')


def imageBrightnessIncrease(self, img, intensity):
    img = np.array(img)
    print(np.max(img))
    print(img.shape)
    ones = np.ones((img.shape[0],img.shape[0]))
    brightness_increase = ones*intensity
    increased = img-brightness_increase
    if np.min(increased)<0:
        decreased = np.clip(increased, -1, 1)
    return np.array(increased, dtype='float')

# Add gaussian blur to images.
def gaussianBlur(self, img, Intensity=1):
    ImageArray = img
    kernel = np.ones((3, 3), np.float32)
    kernel[0] = [1,2,1]
    kernel[1] = [2,4,2]
    kernel[2] = kernel[0]
    kernel = (kernel * Intensity)/16
    blurred = cv2.filter2D(ImageArray, -1, kernel)
    return blurred

# Randomly crop out a section of an image dependng on the requred size
def getRandomCrop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop