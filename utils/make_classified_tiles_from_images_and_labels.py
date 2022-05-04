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
## This function will take an directory of images and their associated labels, create image tiles, and sort them into folders based on class

from joblib import Parallel, delayed
from glob import glob
import numpy as np 
from imageio import imread, imwrite
import sys, getopt, os
from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import random, string
from scipy.stats import mode as md

from tkinter import Tk#, Toplevel 
from tkinter.filedialog import askopenfilename
# import tkinter
# import tkinter as tk
from tkinter.messagebox import *   
from tkinter.filedialog import *
# import os.path as path

# =========================================================
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')

# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      fp = outpath+os.sep+labels[l]+os.sep+outfile
      imwrite(fp, tmp)

#==============================================================
if __name__ == '__main__':

   direc = ''; tile = ''; thres = ''; 

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"ht:a:b:")
   except getopt.GetoptError:
      print('python make_classified_tiles_from_images_and_labels.py -t tilesize -a threshold')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: make_classified_tiles_from_images_and_labels.py -t 96 -a 0.9')
         sys.exit()
      elif opt in ("-t"):
         tile = arg
      elif opt in ("-a"):
         thres = arg
		 
   if not direc:
      direc = 'train'
   if not tile:
      tile = 256
   if not thres:
      thres = .7
	  
   tile = int(tile)
   thres = float(thres)

   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   labfiles = sorted(askopenfilename(filetypes=[("pick label files","*.jpg *.png")], multiple=True)  )

   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   imfiles = sorted(askopenfilename(filetypes=[("pick image files","*.jpg *.png")], multiple=True)  )
      
   #=======================================================
   direc = imdirec = os.path.dirname(labfiles[0])##'useimages'
   outpath = direc+os.sep+'tile_'+str(tile)

   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
   classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

   with open(classfile) as f:
      classes = f.readlines()

   labels = [l.strip() for l in classes]
   #=======================================================

   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in labels:
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass
   #=======================================================

   #=======================================================
   for f,fim in zip(labfiles,imfiles):
      print('Working on %s' % f)
      try:
         dat = np.array(imread(f))
         print('Generating tiles from dense class map ....')
         Z,ind = sliding_window(imread(fim), (tile,tile,3), (int(tile/2), int(tile/2),3)) 

         C,ind = sliding_window(dat, (tile,tile), (int(tile/2), int(tile/2))) 

         w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath, thres) for k in range(len(Z))) 
      except:
          pass 
		 

