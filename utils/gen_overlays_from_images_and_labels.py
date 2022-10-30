# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ##========================================================

# allows loading of functions from the src directory
import sys, os, getopt, shutil
sys.path.insert(1, '../')
# from annotations_to_segmentations import *
# from image_segmentation import *
from doodler_engine.annotations_to_segmentations import *
from doodler_engine.image_segmentation import *


from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import plotly.express as px
import matplotlib

from numpy.lib.npyio import load
from joblib import delayed, Parallel
from PIL import Image, ImageDraw, ExifTags


###===========================================================
try:
    from my_defaults import *
    print("Your session defaults loaded")
except:
    from defaults import *

###===========================================================
def make_dir(dirname):
    # check that the directory does not already exist
    if not os.path.isdir(dirname):
        # if not, try to create the directory
        try:
            os.mkdir(dirname)
        # if there is an exception, print to screen and try to continue
        except Exception as e:
            print(e)
    # if the dir already exists, let the user know
    else:
        print('{} directory already exists'.format(dirname))

###===========================================================
def move_files(files, outdirec):
    for a_file in files:
        shutil.move(a_file, outdirec+os.sep+a_file.split(os.sep)[-1])

###===========================================================
def print_jpeg(l, i, classes):
    NCLASSES  = len(classes)

    print("Working on %s" % (l))
    lab = np.round(io.imread(l, as_gray=True))
    lab = Image.fromarray(lab)
    # im = io.imread(i)[:,:,0]
    image = Image.open(i)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
            lab=lab.rotate(180, expand=True)
            print('rotating {} by 180 deg'.format(i))
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
            lab=lab.rotate(270, expand=True)
            print('rotating {} by 270 deg'.format(i))
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
            print('rotating {} by 90 deg'.format(i))
            lab=lab.rotate(90, expand=True)

        if 'jpg' in i:
            image.save(i.replace('.jpg','_image.jpg'), format='JPEG')	
            lab.save(l.replace('_label.jpg','_label_use.jpg'), format='JPEG')	
        elif 'jpeg' in i:
             image.save(i.replace('.jpeg','_image.jpeg'), format='JPEG')	
             lab.save(l.replace('_label.jpeg','_label_use.jpeg'), format='JPEG')
        else:
            print("error: {}".format(i))


        im = np.array(image)[:,:,0]
        lab = np.array(lab)

    except(AttributeError, KeyError, IndexError):
        print("No exif data")
        im = io.imread(i)[:,:,0]

        if 'jpg' in i:
            image.save(i.replace('.jpg','_image.jpg'), format='JPEG')	
            lab.save(l.replace('_label.jpg','_label_use.jpg'), format='JPEG')	
        elif 'jpeg' in i:
             image.save(i.replace('.jpeg','_image.jpeg'), format='JPEG')	
             lab.save(l.replace('_label.jpeg','_label_use.jpeg'), format='JPEG')
        else:
            print("error: {}".format(i))


    class_label_names = [c.strip() for c in classes]

    NUM_LABEL_CLASSES = len(class_label_names)

    if NUM_LABEL_CLASSES<=10:
        class_label_colormap = px.colors.qualitative.G10
    else:
        class_label_colormap = px.colors.qualitative.Light24

    # we can't have fewer colors than classes
    assert NUM_LABEL_CLASSES <= len(class_label_colormap)

    cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES+1])

    #Make an overlay
    plt.imshow(im)
    plt.imshow(lab, cmap=cmap, alpha=0.6, vmin=0, vmax=NCLASSES)
    plt.axis('off')
    if 'jpg' in i:
        plt.savefig(i.replace('.jpg','_overlay.png'), dpi=200, bbox_inches='tight')
    elif 'jpeg' in i:
        plt.savefig(i.replace('.jpeg','_overlay.png'), dpi=200, bbox_inches='tight')
    else:
        plt.close()


###===========================================================
def make_jpegs():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of images', initialdir=os.getcwd())
    image_files = sorted(glob(direc+'/*.jp*g'))

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    label_direc = askdirectory(title='Select directory of label images', initialdir=direc)
    label_files = sorted(glob(label_direc+'/*.jp*g'))

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    classfile = askopenfilename(title='Select file containing class (label) names', initialdir=label_direc, filetypes=[("Pick classes.txt file","*.txt")])

    with open(classfile) as f:
        classes = f.readlines()
    
    classes_list = [classes for k in label_files]

    print('Found {} image and {} label files'.format(len(image_files),len(label_files)))

    Parallel(n_jobs=-2)(delayed(print_jpeg)(l,i,c) for l,i,c in zip(label_files,image_files, classes_list))

    # for l,i,c in zip(label_files,image_files, classes_list):
    #     print_jpeg(l,i,c)

    overdir = os.path.join(direc, 'overlays')
    make_dir(overdir)

    ovfiles = glob(direc+'/*_overlay.png')
    outdirec = os.path.normpath(direc + os.sep+'overlays')
    move_files(ovfiles, outdirec)

    overdir = os.path.join(direc, 'images')
    make_dir(overdir)

    ovfiles = glob(direc+'/*_image.jpg')
    outdirec = os.path.normpath(direc + os.sep+'images')
    move_files(ovfiles, outdirec)

    ovfiles = glob(direc+'/*_image.jpeg')
    outdirec = os.path.normpath(direc + os.sep+'images')
    move_files(ovfiles, outdirec)

    overdir = os.path.join(label_direc, 'labels')
    make_dir(overdir)

    ovfiles = glob(label_direc+'/*label_use.jpg')
    outdirec = os.path.normpath(label_direc + os.sep+'labels')
    move_files(ovfiles, outdirec)

    ovfiles = glob(label_direc+'/*label_use.jpeg')
    outdirec = os.path.normpath(label_direc + os.sep+'labels')
    move_files(ovfiles, outdirec)


###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:") 
    except getopt.GetoptError:
        print('======================================')
        print('python gen_overlays_from_images_and_labels.py') 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python gen_overlays_from_images_and_labels.py') 
            print('======================================')
            sys.exit()
    #ok, dooo it
    make_jpegs()


# boom.    