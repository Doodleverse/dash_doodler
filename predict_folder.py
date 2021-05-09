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
import os, sys, getopt
from glob import glob
from datetime import datetime

from image_segmentation import * #extract_features, crf_refine, rescale, filter_one_hot
from annotations_to_segmentations import img_to_ubyte_array, label_to_colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.io import imsave
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from tempfile import TemporaryFile

## user defined parameters
##=========================================================

try:
    from my_defaults import *
    print("Your session defaults loaded")
except:
    from defaults import *

###===========================================================
def tta_crf(img, rf_result_filt_inp, k):
    k = int(k)
    result2, n = crf_refine(np.roll(rf_result_filt_inp,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB)
    result2 = np.roll(result2, -k)
    if k==0:
        w=.1
    else:
        w = 1/np.sqrt(k)

    if len(np.unique(result2))>1:
        result2 = filter_one_hot(result2, 2*result2.shape[0])
    else:
        #print('crf refine failed')
        result2 = rf_result_filt_inp.copy()

    return result2, w,n


## function that does everything
##=========================================================
def segment_images(save_mode,do_plot,print_labels,distance):

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    RF_model_file = askopenfilename(title='Select Random Forest classifier file', filetypes=[("Pick classifier file","*.z")])

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

    results_folder = 'results/pred_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    try:
        os.mkdir(results_folder)
        print("Results will be written to %s" % (results_folder))
    except:
        pass

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of image samples', initialdir=os.getcwd())
    files = sorted(glob(direc+os.sep+'*.jpg')) + sorted(glob(direc+os.sep+'*.JPG')) + sorted(glob(direc+os.sep+'*.jpeg'))

    multichannel = True
    intensity = True
    texture = True
    edges = True
    clf = load(RF_model_file) #load RF model from file

    ### main loop thru files
    for file in tqdm(files):
        print("Working on %s" % (file))
        img = img_to_ubyte_array(file) # read image into memory

        Horig = img.shape[0]
        Worig = img.shape[1]

        ##=====features
        #print("Extracting features")

        savez_dict = dict()

        # #standardization using adjusted standard deviation
        # N = np.shape(img)[0] * np.shape(img)[1]
        # s = np.maximum(np.std(img), 1.0/np.sqrt(N))
        # m = np.mean(img)
        # img = (img - m) / s
        # img = rescale(img, 0, 1)
        # del s, m, N
        #
        # if np.ndim(img)!=3:
        #     img = np.dstack((img,img,img))

        img = standardize(img)

        # outfile = TemporaryFile()
        # dtype = img.dtype
        # shape = img.shape
        # fp = np.memmap(outfile, dtype=dtype, mode='w+', shape=shape)
        # fp[:] = img[:]
        # fp.flush()
        # del img
        # del fp
        #
        # img = np.memmap(outfile, dtype=dtype, mode='r', shape=shape)

        if np.ndim(img)==3:
            features = extract_features(
                img,
                multichannel=multichannel,
                intensity=intensity,
                edges=edges,
                texture=texture,
                sigma_min=SIGMA_MIN,
                sigma_max=SIGMA_MAX,
            )
        else:
            features = extract_features(
                np.dstack((img,img,img)),
                multichannel=multichannel,
                intensity=intensity,
                edges=edges,
                texture=texture,
                sigma_min=SIGMA_MIN,
                sigma_max=SIGMA_MAX,
            )

        outfile2 = TemporaryFile()
        dtype2 = features.dtype
        shape2 = features.shape
        fp = np.memmap(outfile2, dtype=dtype2, mode='w+', shape=shape2)
        fp[:] = features[:]
        fp.flush()
        del features
        del fp
        features = np.memmap(outfile2, dtype=dtype2, mode='r', shape=shape2)

        ##=====RF

        # counter = 1
        # for k in features:
        #     plt.subplot(15,6,counter)
        #     plt.imshow(k, cmap='gray')
        #     plt.axis('off')
        #     counter +=1
        #print("RF model ")

        # use model in predictive mode
        sh = features.shape
        features = features.reshape((sh[0], np.prod(sh[1:]))).T

        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)

        rf_result = clf.predict(features)

        if save_mode:
            savez_dict['features'] = features.astype('float16')

        del features

        #print("One-hot filtering ")

        rf_result = rf_result.reshape(sh[1:])
        if save_mode:
            savez_dict['rf_result'] = rf_result

        rf_result_filt = filter_one_hot(rf_result, 2*rf_result.shape[0])
        del rf_result
        if save_mode:
            savez_dict['rf_result_filt'] = rf_result_filt

        #print("Spatial filtering ")

        rf_result_filt = filter_one_hot_spatial(rf_result_filt, distance)
        if save_mode:
            savez_dict['rf_result_spatfilt'] = rf_result_filt

        rf_result_filt = rf_result_filt.astype('float')
        rf_result_filt[rf_result_filt==0] = np.nan
        rf_result_filt_inp = inpaint_nans(rf_result_filt).astype('uint8')

        if save_mode:
            savez_dict['rf_result_filt'] = rf_result_filt
            savez_dict['rf_result_filt_inp'] = rf_result_filt_inp

        del rf_result_filt

        ##=====CRF
        if save_mode:
            savez_dict['image'] = (255*img).astype('uint8')
        else:
            savez_dict['arr_0'] = (255*img).astype('uint8') #for segmentation zoo


        if len(np.unique(rf_result_filt_inp))==1:

            if save_mode:
                savez_dict['crf_tta'] = None
                savez_dict['crf_tta_weights'] = None
                savez_dict['crf_result'] =None
                savez_dict['rf_result_spatfilt'] = None
                savez_dict['crf_result_filt'] = None
                savez_dict['final_label'] = rf_result_filt_inp-1
            else:
                savez_dict['arr_1'] = rf_result_filt_inp-1

            ## npz
            if 'jpg' in file:
                np.savez(file.replace(direc,results_folder).replace('.jpg','_proc.npz'), **savez_dict )
            elif 'jpeg' in file:
                np.savez(file.replace(direc,results_folder).replace('.jpeg','_proc.npz'), **savez_dict )
            elif 'JPG' in file:
                np.savez(file.replace(direc,results_folder).replace('.JPG','_proc.npz'), **savez_dict )

            del savez_dict

        else:

            #print("CRF refinement ")
            del img
            img = img_to_ubyte_array(file) # read image into memory
            #standardization using adjusted standard deviation
            N = np.shape(img)[0] * np.shape(img)[1]
            s = np.maximum(np.std(img), 1.0/np.sqrt(N))
            m = np.mean(img)
            img = (img - m) / s
            img = rescale(img, 0, 1)
            del s, m, N

            if np.ndim(img)!=3:
                img = np.dstack((img,img,img))

            w = Parallel(n_jobs=-2, verbose=0)(delayed(tta_crf)(img, rf_result_filt_inp, k) for k in np.linspace(0,int(img.shape[0]),10))
            R,W,n = zip(*w)
            del rf_result_filt_inp

            crf_result = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')

            if save_mode:
                savez_dict['crf_tta'] = [r.astype('uint8') for r in R]
                savez_dict['crf_tta_weights'] = W
            del R, W, n, w

            if save_mode:
                savez_dict['crf_result'] = crf_result-1

            crf_result_filt = filter_one_hot(crf_result, 2*crf_result.shape[0])
            del crf_result
            crf_result_filt = filter_one_hot_spatial(crf_result_filt, distance)
            crf_result_filt = crf_result_filt.astype('float')
            crf_result_filt[crf_result_filt==0] = np.nan
            crf_result_filt_inp = inpaint_nans(crf_result_filt).astype('uint8')-1
            del crf_result_filt

            if save_mode:
                savez_dict['final_label'] = crf_result_filt_inp.astype('uint8')
            else:
                savez_dict['arr_1'] = crf_result_filt_inp.astype('uint8') #for segmentation zoo

            ##=====file outputs

            ## npz
            if 'jpg' in file:
                np.savez(file.replace(direc,results_folder).replace('.jpg','_proc.npz'), **savez_dict )
            elif 'jpeg' in file:
                np.savez(file.replace(direc,results_folder).replace('.jpeg','_proc.npz'), **savez_dict )
            elif 'JPG' in file:
                np.savez(file.replace(direc,results_folder).replace('.JPG','_proc.npz'), **savez_dict )
            del savez_dict

            ## label images ti png
            if print_labels:
                #print("Printing to file ")
                if np.ndim(img)==3:
                    if 'jpg' in file:
                        imsave(file.replace(direc,results_folder).replace('.jpg','_label.png'),
                                label_to_colors(crf_result_filt_inp, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
                    elif 'jpeg' in file:
                        imsave(file.replace(direc,results_folder).replace('.jpeg','_label.png'),
                                label_to_colors(crf_result_filt_inp, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
                    elif 'JPG' in file:
                        imsave(file.replace(direc,results_folder).replace('.JPG','_label.png'),
                                label_to_colors(crf_result_filt_inp, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))


                else:
                    if 'jpg' in file:
                        imsave(file.replace(direc,results_folder).replace('.jpg','_label.png'),
                                label_to_colors(crf_result_filt_inp, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
                    elif 'jpeg' in file:
                        imsave(file.replace(direc,results_folder).replace('.jpeg','_label.png'),
                                label_to_colors(crf_result_filt_inp, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
                    elif 'JPG' in file:
                        imsave(file.replace(direc,results_folder).replace('.JPG','_label.png'),
                                label_to_colors(crf_result_filt_inp, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

                if 'jpg' in file:
                    imsave(file.replace(direc,results_folder).replace('.jpg','_label_greyscale.png'), crf_result_filt_inp)
                elif 'jpeg' in file:
                    imsave(file.replace(direc,results_folder).replace('.jpeg','_label_greyscale.png'), crf_result_filt_inp)
                elif 'JPG' in file:
                    imsave(file.replace(direc,results_folder).replace('.JPG','_label_greyscale.png'), crf_result_filt_inp)

            ## image/label overlay to png
            if do_plot:

                plt.imshow(img); plt.axis('off')
                if np.ndim(img)==3:
                    plt.imshow(label_to_colors(crf_result_filt_inp, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False), alpha=0.4)#, cmap=cmap, vmin=0, vmax=NUM_LABEL_CLASSES)
                else:
                    plt.imshow(label_to_colors(crf_result_filt_inp, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False), alpha=0.4)#, cmap=cmap, vmin=0, vmax=NUM_LABEL_CLASSES)

                if 'jpg' in file:
                    plt.savefig(file.replace(direc,results_folder).replace('.jpg','_overlay.png'), dpi=200, bbox_inches='tight'); plt.close('all')
                elif 'jpeg' in file:
                    plt.savefig(file.replace(direc,results_folder).replace('.jpeg','_overlay.png'), dpi=200, bbox_inches='tight'); plt.close('all')
                elif 'JPG' in file:
                    plt.savefig(file.replace(direc,results_folder).replace('.JPG','_overlay.png'), dpi=200, bbox_inches='tight'); plt.close('all')

            del crf_result_filt_inp, img



###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:m:p:l:d:") #t:
    except getopt.GetoptError:
        print('======================================')
        print('python predict_folder.py [-m save mode -p make image/label overlay plots -l print label images -d distance ]') #
        print('======================================')
        print('Example usage: python predict_folder.py [default -m 1 -p 0 -l 0 -d 3]') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
        print('.... which means: save mode mode 1 (default, minimal), make image/label overlay plots 0 (no), print label images 0 (no), distance = 3 px') #, save mode mode 1 , dont make plots,
        print('Example usage: python predict_folder.py -m 1 -l 1') #, save mode 1 (default, minimal), make plots 0 (no), print labels 1 (print label images to png files)
        print('.... which means: save mode 1 (default, minimal), make image/label overlay plots 0 (no), print label images 1 (print label images to png files)') #, save mode 1 (default, minimal), make plots 0 (no), print labels 1 (print label images to png files)
        print('Example usage: python predict_folder.py -m 2 -p 1 -l 0') # save mode 2 (everything), make plots 1 (yes), print labels 0 (no)
        print('.... which means: save mode 2 (everything), make image/label overlay plots 1 (yes), print label images 0 (no)') # save mode 2 (everything), make plots 1 (yes), print labels 0 (no)
        print('======================================')

        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python predict_folder.py [default -m 1 -p 0 -l 0 -d 3]') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
            print('.... which means: save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no), distance = 3 px') #, save mode mode 1 , dont make plots,
            print('Example usage: python predict_folder.py -m 1 -l 1') #, save mode 1 (default, minimal), make plots 0 (no), print labels 1 (print label images to png files)
            print('.... which means: save mode 1 (default, minimal), make plots 0 (no), print labels 1 (print label images to png files)') #, save mode 1 (default, minimal), make plots 0 (no), print labels 1 (print label images to png files)
            print('Example usage: python predict_folder.py -m 2 -p 1 -l 0') # save mode 2 (everything), make plots 1 (yes), print labels 0 (no)
            print('.... which means: save mode 2 (everything), make plots 1 (yes), print labels 0 (no)') # save mode 2 (everything), make plots 1 (yes), print labels 0 (no)
            print('======================================')
            sys.exit()
        elif opt in ("-p"):
            do_plot = arg
            do_plot = bool(do_plot)
        elif opt in ("-m"):
            save_mode = arg
            save_mode = bool(save_mode)
        elif opt in ("-l"):
            print_labels = arg
            print_labels = bool(print_labels)
        elif opt in ("-d"):
            distance = arg
            distance = int(distance)

    if 'save_mode' not in locals():
        save_mode = True
    if 'do_plot' not in locals():
        do_plot = False
    if 'print_labels' not in locals():
        print_labels = False
    if 'distance' not in locals():
        distance = 3

    print("save mode: %i" % (save_mode))
    print("make plots: %i" % (do_plot))
    print("print label images: %i" % (print_labels))
    print("threshold intra-label distance: %i" % (distance))

    #ok, dooo it
    #distance = 3
    segment_images(save_mode,do_plot,print_labels,distance)


    # R = []; W = []
    # counter = 0
    # for k in np.linspace(0,int(img.shape[0]/5),5):
    #     k = int(k)
    #     result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine
    #
    #     result2 = np.roll(result2, -k)
    #     R.append(result2)
    #     counter +=1
    #     if k==0:
    #         W.append(0.1)
    #     else:
    #         W.append(1/np.sqrt(k))
    #
    # for k in np.linspace(0,int(img.shape[0]/5),5):
    #     k = int(k)
    #     result2, _ = crf_refine(np.roll(result,-k), np.roll(img,-k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine
    #
    #     result2 = np.roll(result2, k)
    #     R.append(result2)
    #     counter +=1
    #     if k==0:
    #         W.append(.1)#np.nan)
    #     else:
    #         W.append(1/np.sqrt(k))

    #W = np.array(W)
    #W[0] = 2*np.nanmax(W)

    # #result2 = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')
    # result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
    #
    # # median filter
    # result = median(result2, disk(DEFAULT_MEDIAN_KERNEL)).astype(np.uint8)


# turn to black and white / binary
###for file in *_greyscale.png; do convert -monochrome $file "${file%label_greyscale.png}mask.jpg"; done


    # R = []
    # for k in np.linspace(0,img.shape[0],5):
    #     k = int(k)
    #     result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine
    #     result2 = np.roll(result2, -k)
    #     R.append(result2)

    #result = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')
