# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
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
import sys
sys.path.insert(1, 'src')
import plotly.express as px
import os
from glob import glob
from datetime import datetime

from image_segmentation import extract_features, crf_refine
from annotations_to_segmentations import img_to_ubyte_array, label_to_colors
import numpy as np
from joblib import load

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.io import imsave
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory


## user defined parameters
##=========================================================

try:
    from my_defaults import *
except:
    from defaults import *
finally:
    DEFAULT_PEN_WIDTH = 2

    DEFAULT_CRF_DOWNSAMPLE = 2

    DEFAULT_RF_DOWNSAMPLE = 10

    DEFAULT_CRF_THETA = 40

    DEFAULT_CRF_MU = 100

    DEFAULT_MEDIAN_KERNEL = 3

    DEFAULT_RF_NESTIMATORS = 3

    DEFAULT_CRF_GTPROB = 0.9

    SIGMA_MIN = 1

    SIGMA_MAX = 16

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
RF_model_file = askopenfilename(filetypes=[("Pick classifier file","*.z")])

##========================================================

with open('classes.txt') as f:
    classes = f.readlines()

class_label_names = [c.strip() for c in classes]

NUM_LABEL_CLASSES = len(class_label_names)


if NUM_LABEL_CLASSES<=10:
    class_label_colormap = px.colors.qualitative.G10
else:
    class_label_colormap = px.colors.qualitative.Light24

cmap = ListedColormap(class_label_colormap)

print(class_label_colormap)

results_folder = 'results/pred_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

try:
    os.mkdir(results_folder)
    print("Results will be written to %s" % (results_folder))
except:
    pass

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
direc = askdirectory(title='Select directory of image samples', initialdir=os.getcwd())
files = sorted(glob(direc+'/*.jpg'))

multichannel = True
intensity = True
texture = True
edges = True
clf = load(RF_model_file) #load RF model from file

# DEFAULT_CRF_MU = 255
# DEFAULT_CRF_THETA = 10

for file in tqdm(files):
    print("Working on %s" % (file))
    img = img_to_ubyte_array(file) # read image into memory

    Horig = img.shape[0]
    Worig = img.shape[1]

    features = extract_features(
        img,
        multichannel=multichannel,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    ) # extract image features

    print("Extracting features")

    # use model in predictive mode
    sh = features.shape
    features = features.reshape((sh[0], np.prod(sh[1:]))).T
    result = clf.predict(features)
    del features
    result = result.reshape(sh[1:])

    print("CRF refinement ")

    # R = []
    # for k in np.linspace(0,img.shape[0],5):
    #     k = int(k)
    #     result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine
    #     result2 = np.roll(result2, -k)
    #     R.append(result2)

    #result = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')

    R = []; W = []
    counter = 0
    for k in np.linspace(0,int(img.shape[0]/5),5):
        k = int(k)
        result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine

        result2 = np.roll(result2, -k)
        R.append(result2)
        counter +=1
        if k==0:
            W.append(0.1)
        else:
            W.append(1/np.sqrt(k))

    for k in np.linspace(0,int(img.shape[0]/5),5):
        k = int(k)
        result2, _ = crf_refine(np.roll(result,-k), np.roll(img,-k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine

        result2 = np.roll(result2, k)
        R.append(result2)
        counter +=1
        if k==0:
            W.append(0.1)
        else:
            W.append(1/np.sqrt(k))

    #result2 = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')
    result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')

    # median filter
    result = median(result, disk(DEFAULT_MEDIAN_KERNEL)).astype(np.uint8)

    print("Printing to file ")

    imsave(file.replace(direc,results_folder).replace('.jpg','_label.png'),
            label_to_colors(result-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

    imsave(file.replace(direc,results_folder).replace('.jpg','_label_greyscale.png'), result)

    plt.imshow(img); plt.axis('off')
    plt.imshow(label_to_colors(result-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False), alpha=0.4)#, cmap=cmap, vmin=0, vmax=NUM_LABEL_CLASSES)
    plt.savefig(file.replace(direc,results_folder).replace('.jpg','_fig.png'), dpi=200, bbox_inches='tight'); plt.close('all')
    del result, img

# turn to black and white / binary
###for file in *_greyscale.png; do convert -monochrome $file "${file%label_greyscale.png}mask.jpg"; done
