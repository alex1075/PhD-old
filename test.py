import cv2
import numpy as np
import glob
import os 
from code.helper.imageTools import *

example_image = np.random.randint(0, 256, (1024, 1024, 3))
img = cv2.imread(example_image)
cv2.imwrite("./uncropped" +'jpeg', img)
random_crop = getRandomCrop(example_image, 64, 64)
omg = cv2.imread(random_crop)
cv2.imwrite("./cropped" + "jpeg", omg)
