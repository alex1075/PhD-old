import cv2
import glob, os
from PIL import Image
from code.utils import *


def convert(path_to_folder='/Users/alexanderhunt/PhD/Data/'):
    for infile in os.listdir(path_to_folder):
        print ("file : " + infile)
        if infile[-3:] == "bmp":
            print ("is bmp")
            outfile = infile[:-3] + "jpg"
            im = Image.open(path_to_folder + infile)
            print ("new filename : " + outfile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-4:] == "tiff":
            print ("is tiff")
            outfile = infile[:-4] + "jpg"
            im = Image.open(path_to_folder + infile)
            out = im.convert("RGB")
            out.save(path_to_folder + outfile, "jpeg", quality=100)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "png":
            print ("is png")
            outfile = infile[:-4] + "jpg"
            img = cv2.imread(path_to_folder + infile)
            cv2.imwrite(path_to_folder + outfile +'jpeg', img)
            os.remove(path_to_folder + infile)
        elif infile[-3:] == "jpg" or infile[-3:] == "jpeg":
            print ("is jpg, no change")
        else:
            print ("Not an image")

#Cycles through iamges in path_to_folder and resize them the desired size
def resizeAllJpg(path_to_folder='/Users/alexanderhunt/PhD/Data/', newhight=342, newwid=342):
  os.chdir(path_to_folder)
  jpgs = glob.glob('./*.jpg' or './*.jpeg')
  for image in jpgs:
      print ("resizing image" + image)
      name_without_extension = os.path.splitext(image)[0]
      img = cv2.imread(image)
      resized, newheight, newwidth = resizeTo(img, newhight, newwid)
      cv2.imwrite(name_without_extension + ".jpg", resized)

#Cycles through videos in path_to_folder and outputs jpg to out_folder
def convertVideoToImage(path_to_folder='/Users/alexanderhunt/PhD/Video/', out_folder='/Users/alexanderhunt/PhD/Data/'):
    for fi in os.listdir(path_to_folder):
        nam, ext = os.path.splitext(fi)
        if ext == '.mp4':
            cam = cv2.VideoCapture(path_to_folder + fi)
            try:
                # creating a folder named data
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
            # if not created then raise error
            except OSError:
                print ('Error: Creating directory of' + out_folder)
            # frame
            currentframe = 0
            while(True):
                # reading from frame
                ret,frame = cam.read()
                if ret:
                    # if video is still left continue creating images
                    name = out_folder + nam + '_frame_' + str(currentframe) + '.jpg'
                    print ('Creating...' + name)
                    # writing the extracted images
                    cv2.imwrite(name, frame)
                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                else:
                    break
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

def normalise(path_to_folder=r'/Users/alexanderhunt/PhD/Data/'):
    os.chdir(path_to_folder)
    jpgs = glob.glob('./*.jpg' or './*.jpeg')
    for infile in jpgs:
        print ("file : " + infile)
        flute = cv2.imread(infile)
        print ("Normalising " + infile)
        im = normaliseImg(flute)
        cv2.imwrite(path_to_folder + infile, im)