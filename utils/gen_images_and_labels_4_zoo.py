# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-2021, Marda Science LLC
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
import sys, os, getopt
sys.path.insert(1, '../app_files/src')
# from annotations_to_segmentations import *
from image_segmentation import *

from glob import glob
import skimage.io as io
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory


###===========================================================
try:
    from my_defaults import *
    print("Your session defaults loaded")
except:
    from defaults import *

###===========================================================
def make_jpegs():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
    files = sorted(glob(direc+'/*.npz'))

    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]


    #### loop through each file
    for counter, anno_file in tqdm(enumerate(files)):

        # print("Working on %s" % (file))
        print("Working on %s" % (anno_file))
        dat = np.load(anno_file)
        data = dict()
        for k in dat.keys():
            try:
                data[k] = dat[k]
            except:
                pass
        del dat


        try:
            classes = data['classes']
        except:
            classes = ['water', 'land']

        NCLASSES  = len(classes)
        class_string = '_'.join([c.strip() for c in classes])

        if 'orig_image' in data.keys():
            im = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
        else:
            im = np.squeeze(data['image'].astype('uint8'))[:,:,:3]

        io.imsave(anno_file.replace('.npz','.jpg'),
                  im, quality=100, chroma_subsampling=False)

        l = np.argmax(data['label'],-1).astype('uint8')+1
        nx,ny = l.shape
        lstack = np.zeros((nx,ny,NCLASSES))
        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == l[...,None]-1).astype(int) #one-hot encode
        l = np.argmax(lstack,-1).astype('uint8')

        io.imsave(anno_file.replace('.npz','_label.jpg'),
                  l, quality=100, chroma_subsampling=False)

        del im


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
