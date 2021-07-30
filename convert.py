import cv2
import glob, os
from PIL import Image
import cv2


def convert(path_to_folder):
    for infile in os.listdir(path_to_folder):
        print ("file : " + infile)
        if infile[-3:] == "bmp":
            print ("is bmp")
            outfile = infile[:-3] + "jpg"
            im = Image.open(path_to_folder + infile)
            print ("new filename : " + outfile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
        elif infile[-4:] == "tiff":
            print ("is tiff")
            outfile = infile[:-4] + "jpg"
            im = Image.open(path_to_folder + infile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
        elif infile[-3:] == "png":
            print ("is png")
            img = cv2.imread(path_to_folder + infile)
            cv2.imwrite(path_to_folder + infile +'jpeg', img)
        elif infile[-3:] == "jpg" or infile[-3:] == "jpeg":
            print ("is jpg, no change")
        else:
            print ("Not an image")




#Grabs biggest dimension and scales the photo so that max dim is now 1280
def resizeTo1280Max(image, inter=cv2.INTER_AREA):
    (height, width) = image.shape[:2]
    if height>width:
        newheight = 1280
        heightratio = height/newheight
        newwidth = int(width/heightratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    elif width>height:
        newwidth = 1280
        widthratio = width/newwidth
        newheight = int(height/widthratio)
        resized = cv2.resize(image, dsize=(newwidth, newheight), interpolation=inter)
        return resized, newheight, newwidth
    else: 
        print('Error')
#Grabs biggest dimension and scales the photo so that max dim is now 1280


#Cycles through all jpg or jpeg in a folder and uses the resizeTo1280Max
def resizeAllJpg(path_to_folder):
  os.chdir(path_to_folder)
  jpgs = glob.glob('./*.jpg' or './*.jpeg')
  for image in jpgs:
      name_without_extension = os.path.splitext(image)[0]
      img = cv2.imread(image)
      resized, newheight, newwidth = resizeTo1280Max(img)
      cv2.imwrite(name_without_extension + ".jpg", resized)

convert('/home/as-hunt/PhD/Data/')

