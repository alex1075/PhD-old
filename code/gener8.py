import numpy as np
import cv2
from PIL import Image
from code.helper.rectangle import Rectangle
from code.helper.transformations import rotateBound

def findPosition(bg_img, top_image="logo.png", scale_range=(5,15), rotation_degree=(-12,12), exclusion_areas=[]):
    """
    Place one image on another image.
    """
    valid = False
    
    #interpolation method for resizing 
    inter = cv2.INTER_AREA

    #get background dimensions
    bg_height, bg_width = bg_img.shape[:2]

    while not valid:
        #read logo
        logo = cv2.imread(top_image)
        #get logo dimensions
        logo_height, logo_width = logo.shape[:2]
        # Find Center of Background Image
        bg_center_x = int(bg_width/2)
        bg_center_y = int(bg_height/2)
        #read logo array with fourth channel 
        logo_array = cv2.imread(top_image, cv2.IMREAD_UNCHANGED)
        #rotate the logo randomly 
        logo_array = rotateBound(logo_array, np.random.randint(rotation_degree[0], rotation_degree[1]))
        # generate random number for random scale
        random_number = np.random.randint(scale_range[0],scale_range[1])
        # Choose either height or width, depending which is smaller to base the logo resize on.
        if bg_height < bg_width:
            new_logo_height = int(bg_height/100 * random_number)
            heightratio = (logo_width/logo_height)
            new_logo_width = int(heightratio*new_logo_height)
            resized_logo = cv2.resize(logo_array, dsize=(
            new_logo_width, new_logo_height), interpolation=inter)
        else:
            new_logo_width = int(bg_width/100 * random_number)
            widthratio = (logo_height/logo_width)
            new_logo_height = int(new_logo_width * widthratio)
            resized_logo = cv2.resize(logo_array, dsize=(
            new_logo_width, new_logo_height), interpolation=inter)

        # Find the absolute value maximum amount the image can be moved
        max_move_x = abs(int(bg_center_x - (new_logo_width/2)))
        max_move_y = abs(int(bg_center_y - (new_logo_height/2)))

        # Create random location to paste the logo into that is within background image and wont make the logo stick out    
        random_move_x = np.random.randint(
            (-max_move_x - (new_logo_width/2)), max_move_x-(new_logo_width/2))
        random_move_y = np.random.randint(
            (-max_move_y - (new_logo_height/2)), max_move_y-(new_logo_width/2))
        x = bg_center_x + random_move_x
        y = bg_center_y + random_move_y
        pos = (x, y)
        containing_area = [
            pos, (pos[0]+new_logo_width, pos[1]+new_logo_height)]
        containing_rect = Rectangle(
            containing_area[0][0], containing_area[1][0], containing_area[0][1], containing_area[1][1])
        if(len(exclusion_areas) == 0):
            break
        # exclusion_areas: [[top_left=(x,y), bot_right=(x,y)], ]
        for area in exclusion_areas:
            print("\t Searching For Valid Areas")
            area_rect = Rectangle(area[0][0], area[1][0], area[0][1], area[1][1])
            # Is in X
            if area_rect.is_intersect(containing_rect):
                print("\t \x1b[1;31m AREAS INTERSECT, LOOKING FOR NEW AREA \x1b[0m")
                valid = False
                break
        else:
            valid = True
            print("\tThere were no intersections!")
    return resized_logo, pos, new_logo_height, new_logo_width, bg_center_x, bg_center_y




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