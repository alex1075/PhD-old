import cv2
import numpy as np


def changePerspective(image, hor_skew_limiter=10, ver_skew_limiter=10):
    """
    Randomly changes perspective of input image. A higher limiter means less freedom to transform.
    A horizontal limiter of 10 means it can transform by 1/10 its width.
    """
    if type(image) == str:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:
        img = image
   
    rows, cols, ch = img.shape
    upper_boundary_cols = cols/hor_skew_limiter
    upper_boundary_rows = rows/ver_skew_limiter

    pts1 = np.float32([[0+np.random.randint(0, upper_boundary_cols), 0+np.random.randint(0, upper_boundary_rows)], [cols-np.random.randint(0, upper_boundary_cols), 0+np.random.randint(0, upper_boundary_rows)],
                       [cols-np.random.randint(0, upper_boundary_cols), rows-np.random.randint(0, upper_boundary_rows)], [0+np.random.randint(0, upper_boundary_cols), rows-np.random.randint(0, upper_boundary_rows)]])
    pts2 = np.float32([[0+np.random.randint(0, upper_boundary_cols), 0+np.random.randint(0, upper_boundary_rows)], [cols-np.random.randint(0, upper_boundary_cols), 0+np.random.randint(0, upper_boundary_rows)],
                       [cols-np.random.randint(0, upper_boundary_cols), rows-np.random.randint(0, upper_boundary_rows)], [0+np.random.randint(0, upper_boundary_cols), rows-np.random.randint(0, upper_boundary_rows)]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))

    return dst 

def rotateBound(image, angle):
    """
    Rotates the input image by the angle (in radians) without cutting off anything. 
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def translateImage(img_path, translate_x, translate_y):
    """
    Takes image or imagepath and translation in x and y and returns
    a translated image. (it will be cut off though so it's not that useful)
    """
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path

    rows, cols = img.shape[:2]

    TranslationMatrix = np.float32(
        [[1, 0, translate_x], [0, 1, translate_y]])
    transformed = cv2.warpAffine(img, TranslationMatrix, (cols, rows))

    return transformed

def noisy(noise_type, image):
    """
    Used for creating random noise in the image.
    More noise types can be added later. 
    Noise types: 
    1 = gaussian 
    2 = blurring2
    3 = salt and pepper
    4 = speckle
    5 = reduce information (average value becomes middle value)
    6 = sharpening
    7 = normal 5x5 blur 
    8 = normal 3x3 blur
    9 = noise removal edges sharp 
    10 = no change 
    """
    if noise_type == 1: #gaussian
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
