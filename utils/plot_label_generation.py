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
import sys, os, getopt
sys.path.insert(1, '../src')
from annotations_to_segmentations import *
from image_segmentation import *

from glob import glob
import skimage.util
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

import matplotlib
import matplotlib.pyplot as plt

from imageio import imwrite

###===========================================================
try:
    sys.path.insert(1, '../')
    from my_defaults import *
    print("Your session defaults loaded")
except:
    DEFAULT_PEN_WIDTH = 3
    DEFAULT_CRF_DOWNSAMPLE = 2
    DEFAULT_RF_DOWNSAMPLE = 2
    DEFAULT_CRF_THETA = 1
    DEFAULT_CRF_MU = 1
    DEFAULT_CRF_GTPROB = 0.9

DEFAULT_RF_NESTIMATORS = 3

###===========================================================
def tta_crf(img, rf_result_filt_inp, k):
    k = int(k)
    result2, n = crf_refine(np.roll(rf_result_filt_inp,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB)
    result2 = np.roll(result2, -k)
    if k==0:
        w=.1
    else:
        w = 1/np.sqrt(k)

    return result2, w,n


###===========================================================
def gen_plot_seq(orig_distance, save_mode):


    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
    files = sorted(glob(direc+'/*.npz'))

    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]

    #### loop through each file
    for anno_file in tqdm(files):


        if os.path.exists(anno_file.replace('.npz','_label.png')):
            print('%s exists ... skipping' % (anno_file.replace('.npz','_label.png')))
            continue
        else:


            # print("Working on %s" % (file))
            print("Working on %s" % (anno_file))
            dat = np.load(anno_file)
            data = dict()
            for k in dat.keys():
                data[k] = dat[k]
            del dat
            # print(data['image'].shape)

            if 'classes' not in locals():

                try:
                    classes = data['classes']
                except:
                    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
                    classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

                    with open(classfile) as f:
                        classes = f.readlines()

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

            cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES])
            cmap2 = matplotlib.colors.ListedColormap(['#000000']+class_label_colormap[:NUM_LABEL_CLASSES])


            savez_dict = dict()

            ## if more than one label ...
            if len(np.unique(data['doodles']))>2:

                # img = data['image']
                # del data['image']

                if 'orig_image' in data.keys():
                    im = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
                else:
                    im = np.squeeze(data['image'].astype('uint8'))[:,:,:3]

                #================================
                ##fig1 - img versus standardized image
                plt.subplot(121)
                plt.imshow(img); plt.axis('off')
                plt.title('a) Original', loc='left', fontsize=7)

                # #standardization using adjusted standard deviation
                img = standardize(img)

                #================================
                ##fig2 - img / doodles
                plt.subplot(122)
                plt.imshow(img); plt.axis('off')
                plt.title('b) Filtered', loc='left', fontsize=7)
                plt.savefig(anno_file.replace('.npz','_image_filt_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()

                tmp = data['doodles'].astype('float')
                tmp[tmp==0] = np.nan

                ## do plot of images and doodles
                plt.imshow(img)
                plt.imshow(tmp, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2) #'inferno')
                plt.axis('off')
                plt.colorbar(shrink=0.5)
                plt.savefig(anno_file.replace('.npz','_image_doodles_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()
                del tmp

                ## "analytical toola" e.g. compute annotations per unit area of image and per class label - is there an ideal number or threshold not to go below or above?

                #####=========================== RF

                if np.ndim(img)==3:
                    features = extract_features(
                        img,
                        multichannel=True,
                        intensity=True,
                        edges=True,
                        texture=True,
                        sigma_min=1, #SIGMA_MIN,
                        sigma_max=16, #SIGMA_MAX,
                    )
                else:
                    features = extract_features(
                        np.dstack((img,img,img)),
                        multichannel=True,
                        intensity=True,
                        edges=True,
                        texture=True,
                        sigma_min=1, #SIGMA_MIN,
                        sigma_max=16, #SIGMA_MAX,
                    )

                counter=1
                for k in [0,1,2,3,4]:
                    plt.subplot(2,5,counter)
                    plt.imshow(features[k].reshape((img.shape[0], img.shape[1])), cmap='gray'); plt.axis('off')
                    if k==0:
                        plt.title('a) Smallest scale', loc='left', fontsize=7)
                    counter+=1

                for k in [70,71,72,73,74]:
                    plt.subplot(2,5,counter)
                    plt.imshow(features[k].reshape((img.shape[0], img.shape[1])), cmap='gray'); plt.axis('off')
                    if k==70:
                        plt.title('b) Largest scale', loc='left', fontsize=7)
                    counter+=1

                plt.savefig(anno_file.replace('.npz','_image_feats_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()

                #================================
                doodles = data['doodles']
                training_data = features[:, doodles > 0].T
                training_labels = doodles[doodles > 0].ravel()
                del doodles

                training_data = training_data[::DEFAULT_RF_DOWNSAMPLE]
                training_labels = training_labels[::DEFAULT_RF_DOWNSAMPLE]

                if save_mode:
                    savez_dict['color_doodles'] = data['color_doodles'].astype('uint8')
                    savez_dict['doodles'] = data['doodles'].astype('uint8')
                    savez_dict['settings'] = data['settings']
                    savez_dict['label'] = data['label'].astype('uint8')

                del data

                #================================
                clf = make_pipeline(
                        StandardScaler(),
                        MLPClassifier(
                            solver='adam', alpha=1, random_state=1, max_iter=2000,
                            early_stopping=True, hidden_layer_sizes=[100, 60],
                        ))
                clf.fit(training_data, training_labels)

                #================================

                del training_data, training_labels

                # use model in predictive mode
                sh = features.shape
                features_use = features.reshape((sh[0], np.prod(sh[1:]))).T

                if save_mode:
                    savez_dict['features'] = features.astype('float16')
                del features

                rf_result = clf.predict(features_use)
                #del features_use
                rf_result = rf_result.reshape(sh[1:])

                #================================
                plt.imshow(img)
                plt.imshow(rf_result-1, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
                plt.axis('off')
                plt.colorbar(shrink=0.5)
                plt.savefig(anno_file.replace('.npz','_image_label_RF_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()

                #================================
                plt.subplot(221); plt.imshow(rf_result-1, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap); plt.axis('off')
                plt.title('a) Original', loc='left', fontsize=7)

                rf_result_filt = filter_one_hot(rf_result, 2*rf_result.shape[0])
                if save_mode:
                    savez_dict['rf_result_filt'] = rf_result_filt

                plt.subplot(222); plt.imshow(rf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                plt.title('b) Filtered', loc='left', fontsize=7)

                if rf_result_filt.shape[0]>512:
                    ## filter based on distance
                    rf_result_filt = filter_one_hot_spatial(rf_result_filt, orig_distance)

                if save_mode:
                    savez_dict['rf_result_spatfilt'] = rf_result_filt

                plt.subplot(223); plt.imshow(rf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                plt.title('c) Spatially filtered', loc='left', fontsize=7)

                # rf_result_filt_inp = inpaint_zeros(rf_result_filt).astype('uint8')

                rf_result_filt = rf_result_filt.astype('float')
                rf_result_filt[rf_result_filt==0] = np.nan
                rf_result_filt_inp = inpaint_nans(rf_result_filt).astype('uint8')

                plt.subplot(224); plt.imshow(rf_result_filt_inp, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                plt.title('d) Inpainted', loc='left', fontsize=7)

                plt.savefig(anno_file.replace('.npz','_rf_label_filtered_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()

                ###========================================================
                #### demo of the spatial filter

                if NUM_LABEL_CLASSES==2:

                    distance = orig_distance #3
                    shrink_factor= 0.66
                    rf_result_filt = filter_one_hot(rf_result, 2*rf_result.shape[0])

                    lstack = (np.arange(rf_result_filt.max()) == rf_result_filt[...,None]-1).astype(int) #one-hot encode

                    plt.figure(figsize=(12,16))
                    plt.subplots_adjust(wspace=0.2, hspace=0.5)

                    plt.subplot(631)
                    plt.imshow(img); plt.imshow(rf_result_filt-1, cmap='gray', alpha=0.25)
                    plt.axis('off');  plt.title('a) Label', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

                    plt.subplot(635)
                    plt.imshow(img); plt.imshow(lstack[:,:,0], cmap='gray', alpha=0.25)
                    plt.axis('off');  plt.title('b) "Zero-hot"', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

                    plt.subplot(636)
                    plt.imshow(img); plt.imshow(lstack[:,:,1], cmap='gray', alpha=0.25)
                    plt.axis('off');  plt.title('c) "One-hot"', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

                    tmp = np.zeros_like(rf_result_filt)
                    for kk in range(lstack.shape[-1]):
                        l = lstack[:,:,kk]
                        d = ndimage.distance_transform_edt(l)
                        l[d<distance] = 0
                        lstack[:,:,kk] = np.round(l).astype(np.uint8)
                        del l
                        tmp[d<=distance] += 1

                        if kk==0:
                            plt.subplot(637)
                            plt.imshow(d, cmap='inferno') # plt.imshow(img);  'gray', alpha=0.5)
                            plt.axis('off'); plt.title('d) Zero-hot distance < '+str(distance)+' px', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);
                        else:
                            plt.subplot(638)
                            plt.imshow(d, cmap='inferno') #'gray', alpha=0.5)
                            plt.axis('off'); plt.title('e) One-hot distance < '+str(distance)+' px', loc='left', fontsize=7) # plt.colorbar(shrink=shrink_factor);
                        del d

                    plt.subplot(6,3,11)
                    plt.imshow(img); plt.imshow(tmp==rf_result_filt.max(), cmap='gray', alpha=0.25)
                    plt.axis('off'); plt.title('f) Distance < threshold (= '+str(distance)+' px)', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

                    rf_result_filt = np.argmax(lstack, -1)+1

                    rf_result_filt[tmp==rf_result_filt.max()] = 0
                    del tmp

                    plt.subplot(6,3,12)
                    plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.25)
                    plt.axis('off'); plt.title('g) Label encoded with zero class', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

                    ##double distance
                    distance *= 3
                    tmp = np.zeros_like(rf_result_filt)
                    for kk in range(lstack.shape[-1]):
                        l = lstack[:,:,kk]
                        d = ndimage.distance_transform_edt(l)
                        l[d<distance] = 0
                        lstack[:,:,kk] = np.round(l).astype(np.uint8)
                        del l
                        tmp[d<=distance] += 1

                        if kk==0:
                            plt.subplot(6,3,13)
                            plt.imshow(d, cmap='inferno') # plt.imshow(img);  'gray', alpha=0.5)
                            plt.axis('off'); plt.title('d) Zero-hot distance < '+str(distance)+' px', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);
                        else:
                            plt.subplot(6,3,14)
                            plt.imshow(d, cmap='inferno') #'gray', alpha=0.5)
                            plt.axis('off'); plt.title('e) One-hot distance < '+str(distance)+' px', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);
                        del d

                    plt.subplot(6,3,17)
                    plt.imshow(img); plt.imshow(tmp==rf_result_filt.max(), cmap='gray', alpha=0.25)
                    plt.axis('off');plt.title('h) Distance < threshold (= '+str(distance)+' px)', loc='left', fontsize=7) # plt.colorbar(shrink=shrink_factor);

                    ###========================================================
                    rf_result_filt = np.argmax(lstack, -1)+1


                    rf_result_filt[tmp==rf_result_filt.max()] = 0
                    del tmp

                    plt.subplot(6,3,18)
                    plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.25)
                    plt.axis('off'); plt.title('i) Label encoded with zero class', loc='left', fontsize=7); #plt.colorbar(shrink=shrink_factor);

                    plt.savefig(anno_file.replace('.npz','_rf_spatfilt_dist_labelgen.png'), dpi=300, bbox_inches='tight')
                    plt.close()

                if save_mode:
                    savez_dict['rf_result'] = rf_result

                del rf_result, rf_result_filt
                if save_mode:
                    savez_dict['rf_result_filt_inp'] = rf_result_filt_inp

                #####=========================== CRF
                if NUM_LABEL_CLASSES==2:
                    # R = W = n = []
                    # for k in np.linspace(0,int(img.shape[0]),10):
                    #     out1, out2, out3 = tta_crf(img, rf_result_filt_inp, k)
                    #     R.append(out1)
                    #     W.append(out2)
                    #     n.append(out3)
                    # this parallel call replaces the above commented out loop
                    w = Parallel(n_jobs=-2, verbose=0)(delayed(tta_crf)(img, rf_result_filt_inp, k) for k in np.linspace(0,int(img.shape[0])/5,10))
                    R,W,n = zip(*w)
                    del rf_result_filt_inp

                    for counter,r in enumerate(R):
                        plt.subplot(5,2,counter+1)
                        plt.imshow(img)
                        plt.imshow(r-1, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
                        plt.axis('off')
                    plt.savefig(anno_file.replace('.npz','_crf_tta_labelgen.png'), dpi=200, bbox_inches='tight')
                    plt.close()

                    if save_mode:
                        savez_dict['crf_tta'] = [r.astype('uint8') for r in R]
                        savez_dict['crf_tta_weights'] = W

                    crf_result = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
                    del R, W, n, w, r

                    #================================
                    plt.subplot(221); plt.imshow(crf_result-1, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap); plt.axis('off')
                    plt.title('a) Original', loc='left', fontsize=7)

                    crf_result_filt = filter_one_hot(crf_result, 2*crf_result.shape[0])

                    if save_mode:
                        savez_dict['crf_result_filt'] = crf_result_filt
                        savez_dict['crf_result'] = crf_result-1

                    del crf_result

                    plt.subplot(222); plt.imshow(crf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                    plt.title('b) Filtered', loc='left', fontsize=7)

                    if crf_result_filt.shape[0]>512:
                        ## filter based on distance
                        crf_result_filt = filter_one_hot_spatial(crf_result_filt, distance)

                    if save_mode:
                        savez_dict['rf_result_spatfilt'] = crf_result_filt

                    plt.subplot(223); plt.imshow(crf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                    plt.title('c) Spatially filtered', loc='left', fontsize=7)

                    crf_result_filt = crf_result_filt.astype('float')
                    crf_result_filt[crf_result_filt==0] = np.nan
                    crf_result_filt_inp = inpaint_nans(crf_result_filt).astype('uint8')
                    del crf_result_filt

                    plt.subplot(224); plt.imshow(crf_result_filt_inp, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                    plt.title('d) Inpainted (final label)', loc='left', fontsize=7)

                    plt.savefig(anno_file.replace('.npz','_crf_label_filtered_labelgen.png'), dpi=200, bbox_inches='tight')
                    plt.close()

                else:

                    if len(np.unique(rf_result_filt_inp.flatten()))>1:

                        crf_result, n = crf_refine(rf_result_filt_inp, img, DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB)

                        #================================
                        plt.subplot(221); plt.imshow(crf_result-1, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap); plt.axis('off')
                        plt.title('a) Original', loc='left', fontsize=7)

                        crf_result_filt = filter_one_hot(crf_result, 2*crf_result.shape[0])

                        if save_mode:
                            savez_dict['crf_result_filt'] = crf_result_filt
                            savez_dict['crf_result'] = crf_result-1

                        del crf_result

                        plt.subplot(222); plt.imshow(crf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                        plt.title('b) Filtered', loc='left', fontsize=7)

                        if crf_result_filt.shape[0]>512:
                            ## filter based on distance
                            crf_result_filt = filter_one_hot_spatial(crf_result_filt, orig_distance)

                        if save_mode:
                            savez_dict['rf_result_spatfilt'] = crf_result_filt

                        plt.subplot(223); plt.imshow(crf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                        plt.title('c) Spatially filtered', loc='left', fontsize=7)

                        #crf_result_filt_inp = inpaint_zeros(crf_result_filt).astype('uint8')
                        crf_result_filt = crf_result_filt.astype('float')
                        crf_result_filt[crf_result_filt==0] = np.nan
                        crf_result_filt_inp = inpaint_nans(crf_result_filt).astype('uint8')
                        del crf_result_filt

                        plt.subplot(224); plt.imshow(crf_result_filt_inp, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
                        plt.title('d) Inpainted (final label)', loc='left', fontsize=7)

                        plt.savefig(anno_file.replace('.npz','_crf_label_filtered_labelgen.png'), dpi=200, bbox_inches='tight')
                        plt.close()
                    else:
                        crf_result_filt_inp = rf_result_filt_inp.copy()

                #================================
                plt.imshow(img)
                plt.imshow(crf_result_filt_inp-1, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
                plt.axis('off')
                plt.colorbar(shrink=0.5)
                plt.savefig(anno_file.replace('.npz','_image_label_final_labelgen.png'), dpi=200, bbox_inches='tight')
                plt.close()


                if save_mode:
                    tosave = (np.arange(crf_result_filt_inp.max()) == crf_result_filt_inp[...,None]-1).astype(int)
                    savez_dict['final_label'] = tosave.astype('uint8')#crf_result_filt_inp-1
                    savez_dict['image'] = (255*img).astype('uint8')
                del img, crf_result_filt_inp

                imwrite(anno_file.replace('.npz','_label.png'), np.argmax(savez_dict['final_label'],-1).astype('uint8'))
                imwrite(anno_file.replace('.npz','_doodles.png'), savez_dict['doodles'].astype('uint8'))


            ### if only one label
            else:

                if 'orig_image' in data.keys():
                    im = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
                else:
                    im = np.squeeze(data['image'].astype('uint8'))[:,:,:3]

                if save_mode:
                    savez_dict['color_doodles'] = data['color_doodles'].astype('uint8')
                    savez_dict['doodles'] = data['doodles'].astype('uint8')
                    savez_dict['settings'] = data['settings']
                    savez_dict['label'] = data['label'].astype('uint8')
                    v = np.unique(data['doodles']).max()#[0]-1
                    if v==2:
                        tmp = np.zeros_like(data['label'])
                        tmp+=1
                    else:
                        tmp = np.ones_like(data['label'])*v
                    tosave = (np.arange(tmp.max()) == tmp[...,None]-1).astype(int)
                    savez_dict['final_label'] = tosave.astype('uint8').squeeze()
                    savez_dict['crf_tta'] = None
                    savez_dict['crf_tta_weights'] = None
                    savez_dict['crf_result'] =None
                    savez_dict['rf_result_spatfilt'] = None
                    savez_dict['crf_result_filt'] = None
                    savez_dict['image'] = im #data['image'].astype('uint8')
                del data

            np.savez(anno_file.replace('.npz','_labelgen.npz'), **savez_dict )
            del savez_dict
            plt.close('all')



###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:d:m:") #m:p:l:")
    except getopt.GetoptError:
        print('======================================')
        print('python plot_label_generation.py [-m save mode -p make image/label overlay plots -l print label images]') #
        print('======================================')
        print('Example usage: python plot_label_generation.py [default -m 1 -p 0]') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
        print('.... which means: save mode mode 1 (default, minimal), make image/label overlay plots 0 (no), print label images 0 (no)') #, save mode mode 1 , dont make plots,
        print('======================================')

        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python plot_label_generation.py [default -m 1 -p 0]') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
            print('.... which means: save mode mode 1 (default, all), make plots 0 (no), print labels 0 (no)') #, save mode mode 1 , dont make plots,
            print('======================================')
            sys.exit()

        elif opt in ("-d"):
            orig_distance = arg
            orig_distance = int(orig_distance)
        elif opt in ("-m"):
            save_mode = arg
            save_mode = bool(save_mode)

    if 'orig_distance' not in locals():
        orig_distance = 2
    if 'save_mode' not in locals():
        save_mode = True

    print("save mode: %i" % (save_mode))
    print("threshold intra-label distance: %i" % (orig_distance))

    #ok, dooo it
    gen_plot_seq(orig_distance, save_mode)
