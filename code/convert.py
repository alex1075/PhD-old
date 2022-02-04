import cv2
import glob, os, datetime
from PIL import Image
from code.helper.utils import *
from code.helper.imageTools import *


def convert(path_to_folder='/Volumes/PhD/PhD/Data/'):
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
def resizeAllJpg(path_to_folder='/Volumes/PhD/PhD/Data/', newhight=1080, newwid=1080):
  jpgs = glob.glob(path_to_folder + '*.jpg')
  for image in jpgs:
      print ("resizing image" + image)
      name_without_extension = os.path.splitext(image)[0]
      img = cv2.imread(image)
      resized, newheight, newwidth = resizeTo(img, newhight, newwid)
      cv2.imwrite(name_without_extension + ".jpg", resized)

#Cycles through videos in path_to_folder and outputs jpg to out_folder
def convertVideoToImage(path_to_folder='/Volumes/PhD/PhD/Video/', out_folder='/Volumes/PhD/PhD/Data/'):
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

def normalise(path_to_folder=r'/Volumes/PhD/PhD/Data/'):
    jpgs = glob.glob(path_to_folder + '*.jpg')
    for infile in jpgs:
        print ("file : " + infile)
        flute = cv2.imread(infile)
        print ("Normalising " + infile)
        im = normaliseImg(flute)
        cv2.imwrite(path_to_folder + infile, im)

def randomCrop(path_to_folder='/Volumes/PhD/PhD/Data/', outfolder='/Volumes/PhD/PhD/Dataset/', crop_height=256, crop_width=256):
    jpgs = glob.glob(path_to_folder + '*.jpg')
    for jpg in jpgs:
        print("randomly cropping: " + jpg)
        flute = cv2.imread(jpg, 0)
        jpg_name = jpg.split('/')[-1]
        infile = jpg_name[:-4] + "_cropped_" + str(datetime.datetime.now()) + ".jpg"
        print(infile)
        im = getRandomCrop(flute, crop_height, crop_width)
        cv2.imwrite(outfolder + infile, im)

def convert2Gray(path_to_folder='/Volumes/PhD/PhD/Dataset/'):
    jpgs = glob.glob(path_to_folder  + '*.jpg')
    for jpg in jpgs:
        print('Converting to grayscale: ' + jpg)
        flute = cv2.imread(jpg, 0)
        print(flute.shape)
        cv2.imwrite(jpg, flute)