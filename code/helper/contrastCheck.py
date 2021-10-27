from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from typing import Tuple



def getROIs(bg_img, logo_img, pos, new_logo_width, new_logo_height, min_contrast_parameter):

    bg_roi = bg_img[pos[1]:pos[1]+new_logo_height,
                    pos[0]:pos[0]+new_logo_width]
    with_logo_roi = logo_img[pos[1]:pos[1] +
                             new_logo_height, pos[0]:pos[0]+new_logo_width]
    return (bg_roi, with_logo_roi)



def findAverageColor(img):
    img = img[:, :, :-1]
    average = img.mean(axis=0).mean(axis=0)
    # avg = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
    return average



def findDominantColors(img, number_of_colors=2):

    pixels = np.float32(img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        pixels, number_of_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant, palette, counts



def plotColors(img):

    dominant, palette, counts = findDominantColors(img)
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)

    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(img)
    ax0.set_title('original')
    ax0.axis('off')

    ax1.imshow(dom_patch)
    ax1.set_title('Dominant colors')
    ax1.axis('off')
    plt.show(fig)



def distance(color1, color2):
    """
    returns the distance between the two colors
    """
    d = np.sqrt(np.square((color1[0]-color2[0])) + np.square(
        (color1[1]-color2[1])) + np.square((color1[1]-color2[1])))
    return d



def isDistanceAnyOk(bg_img, logo, pos, new_logo_width: int, new_logo_height: int, min_distance=1):
    """Check to See If the Contrast is OK"""

    bg_roi = bg_img[pos[1]:pos[1]+new_logo_height,
                    pos[0]:pos[0]+new_logo_width]
    dominant, palette, counts = findDominantColors(bg_roi, number_of_colors=2)

    avg_logo_color = findAverageColor(logo)
    for color in palette:
        print(f'\tLogo Average: {avg_logo_color}')
        print(f'\tColor: {color}')
        dist = distance(color, avg_logo_color)
        if(dist > min_distance):
            return True

    return False


def isContrastAnyOk(bg_img, logo, pos, new_logo_width: int, new_logo_height: int, min_contrast=1):
    """Check to See If the Contrast is OK"""

    bg_roi = bg_img[pos[1]:pos[1]+new_logo_height,
                    pos[0]:pos[0]+new_logo_width]
    dominant, palette, counts = findDominantColors(bg_roi, number_of_colors=2)

    avg_logo_color = findAverageColor(logo)
    for color in palette:
        contrast = get_contrast(avg_logo_color, color)
        contrast2 = get_contrast(color, avg_logo_color)
        if contrast > min_contrast or contrast2 > min_contrast:
            return True

    return False


def isContrastAllOk(bg_img, logo, pos, new_logo_width: int, new_logo_height: int, min_contrast=1):
    """Check to See If the Contrast is OK"""

    bg_roi = bg_img[pos[1]:pos[1]+new_logo_height,
                    pos[0]:pos[0]+new_logo_width]
    dominant, palette, counts = findDominantColors(bg_roi, number_of_colors=2)

    avg_logo_color = findAverageColor(logo)
    for color in palette:
        contrast = get_contrast(color, avg_logo_color)
        contrast2 = get_contrast(avg_logo_color, color)
        if contrast < min_contrast and contrast2 < min_contrast:
            return False
            break
    else:
        return True



def luminanace(color: Tuple[int, int, int]):
    a = []
    for v in color:
        v = v / 255
        if (v <= 0.03928):
            a.append(v / 12.92)
        else:
            a.append(((v + 0.055) / 1.055) ** 2.4)
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722



def get_contrast(rgb1, rgb2):
    color1 = (rgb1[0], rgb1[1], rgb1[2])
    color2 = (rgb2[0], rgb2[1], rgb2[2])
    return (luminanace(color1) + 0.05) / (luminanace(color2) + 0.05)



def test_contast():
    print(f"__Contrast Test__")
    assert get_contrast([255, 255, 255], [
                        255, 255, 0]) == 1.0738392309265699, "Should be 1.074 for Yellow on White"
    assert get_contrast([255, 255, 255], [
                        0, 0, 255]) == 8.592471358428805, "Should be 8.592 for Blue on White"



if __name__ == "__main__":
    test_contast()
    print("Everything passed")