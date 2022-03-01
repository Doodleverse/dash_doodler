# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-2022, Marda Science LLC
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
sys.path.insert(1, '../app_files/src')
# from annotations_to_segmentations import *
from image_segmentation import *

from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import plotly.express as px
import matplotlib

from numpy.lib.npyio import load

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

def move_files(files, outdirec):
    for a_file in files:
        shutil.move(a_file, outdirec+os.sep+a_file.split(os.sep)[-1])


def make_jpegs():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
    files = sorted(glob(direc+'/*.npz'))

    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]

    #### loop through each file
    for counter, anno_file in enumerate(files):

        # print("Working on %s" % (file))
        print("Working on %s" % (anno_file))

        # try:
        #     dat = np.load(anno_file)
        # except:
        #     dat = np.load(anno_file, allow_pickle=True)
        #
        # finally:
        #     print("Could not load"+anno_file)
        #     pass

        data = dict()
        with load(anno_file, allow_pickle=True) as dat:
            #create a dictionary of variables
            #automatically converted the keys in the npz file, dat to keys in the dictionary, data, then assigns the arrays to data
            for k in dat.keys():
                data[k] = dat[k]
            del dat

        # if 'dat' in locals():
        #
        #     data = dict()
        #     for k in dat.keys():
        #         try:
        #             data[k] = dat[k]
        #         except Exception as e:
        #             print(e)
        #             pass
        #     del dat

        try:
            classes = data['classes']
        except:
            # print('No classes found in settings! Using defaults of "water" and "land"')
            # classes = ['water', 'land']
            Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

            with open(classfile) as f:
                classes = f.readlines()

        NCLASSES  = len(classes)
        class_string = '_'.join([c.strip() for c in classes])

        #Make the original images as jpg
        if 'orig_image' in data.keys():
            im = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
        else:
            if data['image'].shape[-1]==4:
                im=np.squeeze(data['image'].astype('uint8'))[:,:,:-1]
                band4=np.squeeze(data['image'].astype('uint8'))[:,:,-1]
            else:
                im = np.squeeze(data['image'].astype('uint8'))[:,:,:3]

        io.imsave(anno_file.replace('.npz','.jpg'),
                  im, quality=100, chroma_subsampling=False)

        if 'band4' in locals():
                io.imsave(anno_file.replace('.npz','_band4.jpg'),
                          band4, quality=100, chroma_subsampling=False)
                del band4


        #Make the label as jpg
        l = np.argmax(data['label'],-1).astype('uint8')+1
        nx,ny = l.shape
        lstack = np.zeros((nx,ny,NCLASSES))
        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == l[...,None]-1).astype(int) #one-hot encode
        l = np.argmax(lstack,-1).astype('uint8')

        io.imsave(anno_file.replace('.npz','_label.jpg'),
                  l, quality=100, chroma_subsampling=False, check_contrast=False)

        # if 'classes' not in locals():
        #
        #     try:
        #         classes = data['classes']
        #     except:
        #         Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        #         classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])
        #
        #         with open(classfile) as f:
        #             classes = f.readlines()

        class_label_names = [c.strip() for c in classes]

        NUM_LABEL_CLASSES = len(class_label_names)

        if NUM_LABEL_CLASSES<=10:
            class_label_colormap = px.colors.qualitative.G10
        else:
            class_label_colormap = px.colors.qualitative.Light24

        # we can't have fewer colors than classes
        assert NUM_LABEL_CLASSES <= len(class_label_colormap)

        colormap = [
            tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
            for h in [c.replace("#", "") for c in class_label_colormap]
        ]

        cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES+1])
        # cmap2 = matplotlib.colors.ListedColormap(['#000000']+class_label_colormap[:NUM_LABEL_CLASSES])

        #Make an overlay
        plt.imshow(im)
        plt.imshow(l, cmap=cmap, alpha=0.5, vmin=0, vmax=NCLASSES)
        plt.axis('off')
        plt.savefig(anno_file.replace('.npz','_overlay.png'), dpi=200, bbox_inches='tight')


        #Make an doodles overlay
        # plt = matplotlib.pyplot
        try:
            doodles = data['doodles'].astype('float')
            doodles[doodles<1] = np.nan
            doodles -= 1
            plt.imshow(im)
            plt.imshow(doodles, cmap=cmap, alpha=0.5, vmin=0, vmax=NCLASSES)
            plt.axis('off')
            plt.savefig(anno_file.replace('.npz','_doodles.png'), dpi=200, bbox_inches='tight')
        except:
            print('no doodles in {}'.format(anno_file))
        del im

        plt.close('all')

    #mk directories for labels and images, to make transition to zoo easy
    imdir = os.path.join(direc, 'images')
    ladir = os.path.join(direc, 'labels')
    overdir = os.path.join(direc, 'overlays')
    doodlesdir = os.path.join(direc, 'doodles')
    make_dir(imdir)
    make_dir(ladir)
    make_dir(overdir)
    make_dir(doodlesdir)

    lafiles = glob(direc+'/*_label.jpg')
    outdirec = os.path.normpath(direc + os.sep+'labels')
    move_files(lafiles, outdirec)

    doodlefiles = glob(direc+'/*_doodles.png')
    outdirec = os.path.normpath(direc + os.sep+'doodles')
    move_files(doodlefiles, outdirec)

    imfiles = glob(direc+'/*.jpg')
    outdirec = os.path.normpath(direc + os.sep+'images')
    move_files(imfiles, outdirec)

    ovfiles = glob(direc+'/*_overlay.png')
    outdirec = os.path.normpath(direc + os.sep+'overlays')
    move_files(ovfiles, outdirec)



###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:") #m:p:l:")
    except getopt.GetoptError:
        print('======================================')
        print('python gen_images_and_labels_4_zoo.py') #
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python gen_images_and_labels_4_zoo.py') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
            print('======================================')
            sys.exit()
    #ok, dooo it
    make_jpegs()
