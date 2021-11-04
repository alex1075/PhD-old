# import the necessary packages
import os


"""
TRAINING CONFIGURATIONS
"""
# initialize batch size. This should be the maximum for your hardware (2 works on a 1080Ti also dependent on image size obviously)
BATCH_SIZE = 2
# initialize number of epochs. This means how many times the data goes through the trainer.
NUM_EPOCHS = 10
# The weights you are using as your beggining weights
COCO_WEIGHTS_PATH = "data/resnet50_coco_best_v2.1.0.h5"


"""""""""""=============""""""""""""""""===============""""""""""""""""""=============================="""""""""""""""""

"""
DATA CREATION CONFIGURATIONS 

Used in create_new_data.py
"""

# PATH
BASE_IMGS_PATH = os.path.abspath("data/base_images")

# PARAMETERS
# number of logos per image
NUM_LOGOS = 4
# minimum contrast: recommended value in the range of 2 - 10
MIN_CONTRAST = 2.0
# how many times you try to find a spot for pasting the logo until you lower the minimum contrast required
MAX_TRIES = 100
# lower contrast by this amount every time max tries has been exceeded
SUBTRACT_CONTRAST = 0.1

# scale logo randomly to % of base image within the scale range
SCALE_RANGE = (5, 15)
# rotate the logo randomly by degrees within the range below
ROTATION_DEGREE = (-1, 1)
# perspective transform limiters, will randomly transform within the limits (higher limits correspond to less skew)
# a hor_skew_limit of 10 corresponds to a maximum perspective transform 1/10th the size of the width of the image being transformed
HOR_SKEW_LIMITER = 20
VER_SKEW_LIMITER = 20

# randomly apply one of these noise transformation to the logo utils.image.transformations
NOISE_RANGE_MIN, NOISE_RANGE_MAX = (1, 7)


"""""""""""=============""""""""""""""""===============""""""""""""""""""=============================="""""""""""""""""

""" 
DATA AUGMENTATION CONFIGURATION 
"""


# iterate through the scaling factors and then create an image/xml scaled to that size
IMAGE_SCALES = [0.6, 0.7, 0.8]
# blur the image by the intensity in [] and save a new blurred version along with a new blurred xml file
BLUR_INTENSITY = [1]
# sharpen the image by the intensity in [] and save a new sharpened version along with a new sharpened xml file
SHARPEN_INTENSITY = [1]
TRANSLATE = [1]  # Do not change
# translate the bndbox label inside the xml file by a random amount between 0 and [] pixels.
SMALL_MOVE_LABEL = [2]
# increase/decrease the size of the bandboxes by [] pixels in each direction
BANDBOX_SIZE_INCREASE_DECREASE = [2]
# rotates the images and the bounding boxes by angle in degrees
ROTATION_ANGLES = [10, 20, 50, 90, 120, 180, 260, 310, 340]