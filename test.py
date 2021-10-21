import cv2
from imageio import volread
from PIL import Image
from skimage import data, io, filters
import numpy as np
import glob
import os 
def normaliseImg(img):
          img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
          return img

path = "/Users/alexanderhunt/PhD/Data/"
# path = r"C:\Users\pc1\Documents\cyclegan\data\originals\HfqrecB_w10 Brightfield_s1_0.TIF"
# name = path[:-4]
# print(name)
# img = cv2.imread(path, -1)
# print(img)
# img = normaliseImg(img)
# print(img)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.show()
# # cv2.imshow("img1", img)
# # cv2.waitKey(0)
# newpath = name+'.png'
# # (ret, thresh) = cv2.threshold(img, 0.00000001, 255, cv2.THRESH_BINARY)
# cv2.imwrite(newpath, img)
# # os.remove(path)

# tif_2_png('C:\\Users\\pc1\\Documents\\cyclegan\\data\\dataset4')



# img = volread(path)
# extension = "png"
# before_extension = path.split('.')[0]
# print(img.shape)
# img = np.array(img, dtype=np.uint8)
# (ret, thresh) = cv2.threshold(img, 0.00000001, 255, cv2.THRESH_BINARY)
# io.imsave(f'{before_extension}.{extension}', thresh)
# print(img[0].shape)
# index = 0
# for each in img:
#     # io.imwrite(f'{before_extension}_{index}.{extension}')
#     print(f'{before_extension}_{index}.{extension}')
#     print(each.shape)
#     index+=1
# exit(0)
def separate_layers(path_to_folder):
    for path in glob.glob(path_to_folder+"/*.jpg"):
        img = volread(path)
        print(img.shape)
        # img = img.astype(np.uint8)
        extension = path.split('.')[-1]
        before_extension = path.split('.')[0]
        img_name = before_extension.split("//")[-1]
        index = 0
        for each in img:
            print(each.shape)
            if not os.path.exists(os.path.join(path_to_folder, str(index))):
                print("created directory")
                os.mkdir(os.path.join(path_to_folder, str(index)))
                print(os.path.join(path_to_folder, str(index)))
            
            savedir = os.path.join(path_to_folder, str(index))
            print('printin savedir')
            print(savedir)
            print('end of savedire')
            io.imsave(os.path.join(savedir, f'{img_name}_{index}.{extension}'), img[index])
            index+=1
    exit(0)


# separate_layers(path)
# savedir = os.path.join(path, str(0))
# print(savedir)
# before_extension = "before_extension"
# index = 0
# extension = 'TIF'
# print(os.path.join(savedir, f'{before_extension}_{index}.{extension}'))
