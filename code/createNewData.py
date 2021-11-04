from PIL import Image, ImageDraw
import os
import glob
import cv2
import numpy as np
from code import config
import sys

from code.helper.image.findPosition import findPosition
from ordino.utils.image.operations import overlayImage
from ordino.utils.image.transformations import noisy, changePerspective
from ordino.utils.image.contrastCheck import isContrastAllOk, isContrastAnyOk
# Delete this:
from ordino.utils.path import buildAbsPath
from ordino.utils.xml.create import createXml
from ordino.utils.path import buildAbsPath

# Currently does not paste the white logo ever. Probably because the contrast for it is not high enough.


def getALogo(path_to_logos_folder):
    logos_list = os.listdir(path_to_logos_folder)
    num_logos = len(logos_list)
    random_index = np.random.randint(0, num_logos)
    # build this path better. from main or something
    path_to_logo = os.path.join(path_to_logos_folder, logos_list[random_index])
    logo_class = logos_list[random_index].split('.')[0][:-1]
    print(f"\tSelected: {logo_class}")
    return (logo_class, path_to_logo)


def pasteLogos(base_image, path_to_logos: str, logos_per_image: int, min_contrast: float = config.MIN_CONTRAST, max_number_of_tries: int = config.MAX_TRIES):
    """
    Takes paths to bg_img and logo and outputs the pasted image along with exclusion areas. 
    Does not make the xml yet. That will be made after all the logos for the given bg_img
    have been pasted. 
    """

    # initialize number of tries to 1
    number_of_tries = 1
    num_logos_pasted = 0
    contrast_is_ok = False
    overlayed_image = []
    exclusion_areas = []
    label_data = []
    while (logos_per_image != num_logos_pasted):
        print(f"\tAttempt: {number_of_tries}")

        if number_of_tries > max_number_of_tries:
            min_contrast -= config.SUBTRACT_CONTRAST
            print(
                f"lowering minimum contrast parameter by {config.SUBTRACT_CONTRAST}")
            max_number_of_tries = max_number_of_tries + max_number_of_tries

        (logo_class, path_to_logo) = getALogo(path_to_logos)

        print("\t exclusion areas before findPosition", exclusion_areas)
        resized_logo, pos, new_logo_height, new_logo_width, bg_center_x, bg_center_y = findPosition(
            base_image, path_to_logo, scale_range=config.SCALE_RANGE, rotation_degree=config.ROTATION_DEGREE, exclusion_areas=exclusion_areas)

        logo = changePerspective(
            resized_logo, hor_skew_limiter=config.HOR_SKEW_LIMITER, ver_skew_limiter=config.VER_SKEW_LIMITER)
        noisy_logo = noisy(np.random.randint(
            config.NOISE_RANGE_MIN, config.NOISE_RANGE_MAX), logo)

        contrast_is_ok = isContrastAllOk(
            base_image, noisy_logo, pos, new_logo_width, new_logo_height, min_contrast=min_contrast)
        if contrast_is_ok:
            print(f"\t \x1b[1;32m IsContrastOk {contrast_is_ok} \x1b[0m")
        else:
            print(f"\t \x1b[1;31m IsContrastOk {contrast_is_ok} \x1b[0m")

        if(contrast_is_ok):
            number_of_tries = 0
            if(len(exclusion_areas) != 0):
                img = overlayImage(overlayed_image, noisy_logo, pos)
            else:
                img = overlayImage(base_image, noisy_logo, pos)
            overlayed_image = img.copy()
            num_logos_pasted += 1
            ymin = pos[1]
            xmin = pos[0]
            ymax = ymin + new_logo_height
            xmax = xmin + new_logo_width
            label_data.append([logo_class, ymin, xmin, ymax, xmax])
            exclusion_area = [
                pos, (pos[0] + new_logo_width, pos[1] + new_logo_height)]

            exclusion_areas.append(exclusion_area)

        number_of_tries += 1

    if(len(overlayed_image) == 0):
        cv2.imshow("base_image", base_image)
        cv2.waitKey(0)
        return base_image, label_data

    return overlayed_image, label_data


def createNewData(path_to_bg_img_folder: str, path_to_logos: str, path_to_save_folder_images: str, path_to_save_folder_annotations: str, set_name: str = "_generated_"):
    """
    Requires logos in project/NAMEOFPROJECT/logos folder to be named by the class name followed by a number: for example: tiktok_text0.png
    """

    PATHS_TO_BG_IMAGES = os.listdir(path_to_bg_img_folder)

    for save_name, path_to_bg_img in enumerate(PATHS_TO_BG_IMAGES):

        path_to_bg_img = os.path.join(path_to_bg_img_folder, path_to_bg_img)
        print(path_to_bg_img)
        bg_img = cv2.imread(path_to_bg_img)
        bg_img_height, bg_img_width = bg_img.shape[:2]

        data = [[bg_img_width, bg_img_height]]

        img, label_data = pasteLogos(
            base_image=bg_img,
            path_to_logos=path_to_logos,
            logos_per_image=config.NUM_LOGOS,
            min_contrast=config.MIN_CONTRAST,
            max_number_of_tries=config.MAX_TRIES)

        data += label_data
        # save the image with pastings to the images folder

        cv2.imwrite(os.path.join(path_to_save_folder_images,
                                 set_name+f"{save_name:05d}.jpg"), img)
        