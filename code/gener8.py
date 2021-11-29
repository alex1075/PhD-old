import numpy as np
import cv2
from PIL import Image
from code.helper.rectangle import Rectangle
from code.helper.finder import findPosition


def generate_ellipses(image, center_coordinates, axesLength, ellipseAngle,color):
    print(axesLength[0], axesLength[1])
    print(center_coordinates)
    # generate small image of 1 ellipse in a black rectangle 
    cv2.ellipse(image, center_coordinates, axesLength, ellipseAngle, color)
    # cv2.imshow('ellipse', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

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
        generate_ellipses(black_image, center_coordinates, axesLength, ellipseAngle, (255,255,255))
        findPosition(black_image, center_coordinates, axesLength, ellipseAngle)    
    return black_image
    cv2.imshow(black_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(black_image, 'black_image' + datetime.datetime.now().strftime +  '.jpg')
    
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


