import numpy as np
import cv2
from PIL import Image
from code.helper.rectangle import Rectangle



def generate_ellipses(random_seed=420):
    # generate small image of 1 ellipse in a black rectangle 
    return "Hey sis!"

def generateBlackImage(random_seed=36):
    black_image = np.zeros([256,256,3],dtype=np.uint8) 
    # Reading an image in default mode
    # Window name in which image is displayed
    # window_name = 'Image' 
    startAngle = 0
    endAngle = 360
    centers_and_dimensions = []
    for i in range(np.random.randint(5,10)):
        axesLength = (np.random.randint(60,75), np.random.randint(60,75))
        ellipseAngle = (np.random.randint(startAngle,endAngle))
        print(axesLength[0], axesLength[1])
        center_coordinates = (np.random.randint(0 + axesLength[0] / 2, 256 - axesLength[0] / 2), np.random.randint(0 + axesLength[1] / 2 , 256 - axesLength[1] / 2))
        centers_and_dimensions.append([axesLength, center_coordinates, ellipseAngle])

    # # Red color in BGR
    # color = (0)  
    # # Line thickness of 5 px
    # thickness = 5
    # # Using cv2.ellipse() method
    # # Draw a ellipse with red line borders of thickness of 5 px
    # image = cv2.ellipse(image, center_coordinates, axesLength,
    #        angle, startAngle, endAngle, color, thickness)
    # # Displaying the image 
    # cv2.imshow(window_name, image) 