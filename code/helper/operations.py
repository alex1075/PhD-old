import cv2
import numpy as np
import os
import glob
from PIL import Image, ImageDraw
from typing import List, Tuple


def overlayImage(bg_img: List[List[int]] , img_overlay, pos):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.
    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    
    You can add alpha mask as an input variable if you wish to blend
    the image in a different way than just pasting it completely 
    onto the background image. 
    """
    bg_img_copy = bg_img.copy()
    #change the img_overlay to the correct format for dealing with PNG with alpha channel
    alpha_mask = img_overlay[:, :,3]/255
    #alpha mask
    img_overlay = img_overlay[:, :, 0:3]
    
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(bg_img_copy.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(bg_img_copy.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], bg_img_copy.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], bg_img_copy.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = bg_img_copy.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        bg_img_copy[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * bg_img_copy[y1:y2, x1:x2, c])

    return bg_img_copy


def resizeImage(image_path: str, scale: float):
    if type(image_path) == str:
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        resized = cv2.resize(
            img, (int(scale*width), int(scale*height)), interpolation=cv2.INTER_CUBIC)
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return resized.astype(int)  # and return xml resized

    else:
        print('image_path is not a string')

def png2jpg(img_path: str):
    img = cv2.imread(img_path)
    name=img_path.split("/")[-1].split(".")[0]
    print(name)

    cv2.imwrite(f"/home/marcin/Documents/Ordino/retina-net/projects/bacter-ai/images/{name}.jpg", img)







def saveImage(path: str, filename: str, image: List[List[int]]):
    
    # newheight, newwidth = resized_image.shape[:2]
    cv2.imwrite(os.path.join(path, filename) , image)


def changeColorsPNG(img_path: str, savename: str ="blurred_text_blackish.png" , pick_color=np.array([0, 0, 0, 255]), change_to_color=np.array([np.random.randint(0, 20), np.random.randint(0, 20), np.random.randint(0, 20), 255])):
    """Slow function which iterates through each pixel in the img_path PNG image and changes pick_color
    to change_to_color and saves the modified image with a new name savename in the folder specified on
    the second to last line. Used mostly to change logo colors such that """
    image = cv2.imread(img_path, -1)
    # cv2.imshow('image', image)

    for undex in range(len(image)):
        for index in range(len(image[undex])):
            if np.array_equal(image[undex][index], (pick_color)):
                print("pixel changed from: ", image[undex][index], " to: ", np.array(
                    [np.random.randint(0, 20), np.random.randint(0, 20), np.random.randint(0, 20), 255]))
                image[undex][index] = np.array([np.random.randint(
                    0, 20), np.random.randint(0, 20), np.random.randint(0, 20), 255])

    # os.chdir('C:\\Users\\mkedz\\Documents\\Ordino\\Technical\\Slash2Esports\\Image_Processing\\trust_gaming_logos\\')
    cv2.imwrite(savename, image)


def addAlphaChannel(imgPath):
    "Take a JPG and make it into a PNG with an alpha channel"
    img = cv2.imread(imgPath)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*255
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    folder, filename = os.path.split(imgPath)
    print(folder, filename)
    os.chdir(folder)
    cv2.imwrite(filename, img_BGRA)

def printChannels(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(img)
    print(b, g, r, a)


if __name__ == "__main__":

    # for imgname in glob.glob("/home/marcin/Documents/Ordino/retina-net/projects/bacterial/images/*"):
    #     png2jpg(imgname)
    #     cv2.waitKey(0)
    #     print(imgname)

   for imgname in glob.glob("/home/marcin/Documents/Ordino/retina-net/projects/bacter-ai/images/*.png"):
        png2jpg(imgname)
        print(f"converted {imgname}")
        os.remove(imgname)
        print(f"removed {imgname}")