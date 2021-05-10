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

from sklearn.preprocessing import StandardScaler

###===========================================================
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


###===========================================================
def gen_plot_seq(orig_distance, save_mode):

    with open('../classes.txt') as f:
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


    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
    files = sorted(glob(direc+'/*.npz'))

    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]

    data_file = 'tmp.npz'
    rf_file = 'tmp.pkl.z'

    #### loop through each file
    for anno_file in tqdm(files):

        # print("Working on %s" % (file))
        print("Working on %s" % (anno_file))
        dat = np.load(anno_file)
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

        savez_dict = dict()

        ## if more than one label ...
        if len(np.unique(data['doodles']))>2:

            img = data['image']
            del data['image']

            #================================
            ##fig1 - img versus standardized image
            plt.subplot(121)
            plt.imshow(img); plt.axis('off')
            plt.title('a) Original', loc='left', fontsize=7)

            # #standardization using adjusted standard deviation
            # N = np.shape(img)[0] * np.shape(img)[1]
            # s = np.maximum(np.std(img), 1.0/np.sqrt(N))
            # m = np.mean(img)
            # img = (img - m) / s
            # img = rescale(img, 0, 1)
            # del m, s, N
            #
            # if np.ndim(img)!=3:
            #     img = np.dstack((img,img,img))

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
            plt.imshow(tmp, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2) #'inferno')
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
            do_sim = False #True
            if do_sim:
                try: #first time around
                    file_training_data, file_training_labels = load(data_file)
                    training_data = np.concatenate((file_training_data, training_data))
                    training_labels = np.concatenate((file_training_labels, training_labels))
                except:
                    pass

                try: #first time around
                    os.remove(data_file)
                except:
                    pass

                dump((training_data, training_labels), data_file, compress=True) #save new file
                try: #first time around
                    clf = load(rf_file) #load last model from file
                except:
                    clf = RandomForestClassifier(n_estimators=DEFAULT_RF_NESTIMATORS, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)

                try: #first time around
                    os.remove(rf_file)
                except:
                    pass

                clf = RandomForestClassifier(n_estimators=DEFAULT_RF_NESTIMATORS, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)

                #scaler = StandardScaler()
                #training_data = scaler.fit_transform(training_data)

                clf.fit(training_data, training_labels)

                dump(clf, rf_file, compress=True) #save new file

            else:

                clf = RandomForestClassifier(n_estimators=DEFAULT_RF_NESTIMATORS, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)

                scaler = StandardScaler()
                training_data = scaler.fit_transform(training_data)

                clf.fit(training_data, training_labels)

            #================================
            plt.figure(figsize=(12,12))
            plt.subplots_adjust(hspace=0.5)
            counter = 1
            loc='abcdefghijklmnop'
            for pair in [ (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4) ]:
                clf2 = RandomForestClassifier(n_estimators=DEFAULT_RF_NESTIMATORS, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)
                clf2.fit(training_data[:,pair], training_labels)

                # Now plot the decision boundary using a fine mesh as input to a
                # filled contour plot
                plot_step = .05
                x_min, x_max = training_data[:, pair[0]].min() - 2, training_data[:, pair[0]].max() + 2
                y_min, y_max = training_data[:, pair[1]].min() - 2, training_data[:, pair[1]].max() + 2
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                     np.arange(y_min, y_max, plot_step))

                #visualize rf decision surface
                ax=plt.subplot(3,3,counter)
                Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=cmap)
                plt.title(loc[counter-1]+')', loc='left', fontsize=8)
                plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels,
                            cmap=cmap,edgecolor='k', s=5, lw=.25)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(7)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(7)
                #plt.axis('off')
                counter+=1

            #plt.show()
            plt.savefig(anno_file.replace('.npz','_RFdecsurf_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            #================================

            del training_data, training_labels

            # use model in predictive mode
            sh = features.shape
            features_use = features.reshape((sh[0], np.prod(sh[1:]))).T

            if not do_sim:
                features_use = scaler.fit_transform(features_use)

            if save_mode:
                savez_dict['features'] = features.astype('float16')
            del features

            rf_result = clf.predict(features_use)
            #del features_use
            rf_result = rf_result.reshape(sh[1:])
            #
            # #first two features only (location and intensity)
            # rf_result2 = clf2.predict(features_use[:,:2])
            # #del features_use
            # rf_result2 = rf_result2.reshape(sh[1:])

            #================================
            #visualize rf feature importances
            #Feature importances are provided by the fitted attribute feature_importances_
            #and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
            importances = clf.feature_importances_

            plt.figure(figsize=(8,12))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.subplot(3, 1, 1)
            plt.bar(np.arange(len(importances)), importances)
            for f in np.argsort(importances)[-4:]:
                plt.axvline(x=f, ymin=0, ymax=1, color='r', linestyle='--')
            plt.ylabel("Feature importance (non-dim.)") #Mean decrease in impurity
            plt.xlabel("Feature")
            plt.title('a)',loc='left', fontsize=7)

            counter=3
            syms='bcdefghijk'
            for f in np.argsort(importances)[-4:]:
                plt.subplot(3,2,counter)
                plt.imshow(features_use[:,f].reshape(sh[1:]), cmap='gray')
                plt.axis('off'); plt.title(syms[counter-3]+') Feature '+str(f),loc='left', fontsize=7)
                counter+=1

            plt.savefig(anno_file.replace('.npz','_RF_featimps_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()
            #================================

            # imsave(anno_file.replace('.npz','_label_RF_col_labelgen.png'), label_to_colors(rf_result-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False), check_contrast=False)

            #================================
            plt.imshow(img)
            plt.imshow(rf_result-1, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
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
            ## if more than one label ...
            # if len(np.unique(rf_result_filt_inp))>1:

            distance = orig_distance #3
            shrink_factor= 0.66
            rf_result_filt = filter_one_hot(rf_result, 2*rf_result.shape[0])

            lstack = (np.arange(rf_result_filt.max()) == rf_result_filt[...,None]-1).astype(int) #one-hot encode

            plt.figure(figsize=(12,16))
            plt.subplots_adjust(wspace=0.2, hspace=0.5)

            plt.subplot(631)
            plt.imshow(img); plt.imshow(rf_result_filt-1, cmap='gray', alpha=0.5)
            plt.axis('off');  plt.title('a) Label', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

            plt.subplot(635)
            plt.imshow(img); plt.imshow(lstack[:,:,0], cmap='gray', alpha=0.5)
            plt.axis('off');  plt.title('b) "Zero-hot"', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

            plt.subplot(636)
            plt.imshow(img); plt.imshow(lstack[:,:,1], cmap='gray', alpha=0.5)
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
            plt.imshow(img); plt.imshow(tmp==rf_result_filt.max(), cmap='gray', alpha=0.5)
            plt.axis('off'); plt.title('f) Distance < threshold (= '+str(distance)+' px)', loc='left', fontsize=7) #plt.colorbar(shrink=shrink_factor);

            rf_result_filt = np.argmax(lstack, -1)+1

            # plt.subplot(438)
            # plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.5)
            # plt.axis('off'); plt.colorbar(shrink=0.25); plt.title('g) Filtered label', loc='left', fontsize=7)

            rf_result_filt[tmp==rf_result_filt.max()] = 0
            del tmp

            plt.subplot(6,3,12)
            plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.5)
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
            plt.imshow(img); plt.imshow(tmp==rf_result_filt.max(), cmap='gray', alpha=0.5)
            plt.axis('off');plt.title('h) Distance < threshold (= '+str(distance)+' px)', loc='left', fontsize=7) # plt.colorbar(shrink=shrink_factor);

            ###========================================================
            rf_result_filt = np.argmax(lstack, -1)+1

            # plt.subplot(4,3,11)
            # plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.5)
            # plt.axis('off'); plt.colorbar(shrink=0.5)

            rf_result_filt[tmp==rf_result_filt.max()] = 0
            del tmp

            plt.subplot(6,3,18)
            plt.imshow(img); plt.imshow(rf_result_filt, cmap='gray', alpha=0.5)
            plt.axis('off'); plt.title('i) Label encoded with zero class', loc='left', fontsize=7); #plt.colorbar(shrink=shrink_factor);

            plt.savefig(anno_file.replace('.npz','_rf_spatfilt_dist_labelgen.png'), dpi=300, bbox_inches='tight')
            plt.close()

            if save_mode:
                savez_dict['rf_result'] = rf_result

            del rf_result, rf_result_filt
            if save_mode:
                savez_dict['rf_result_filt_inp'] = rf_result_filt_inp

            #####=========================== CRF

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
                plt.imshow(r-1, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
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

            #crf_result_filt_inp = inpaint_zeros(crf_result_filt).astype('uint8')
            crf_result_filt = crf_result_filt.astype('float')
            crf_result_filt[crf_result_filt==0] = np.nan
            crf_result_filt_inp = inpaint_nans(crf_result_filt).astype('uint8')
            del crf_result_filt

            plt.subplot(224); plt.imshow(crf_result_filt_inp, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
            plt.title('d) Inpainted (final label)', loc='left', fontsize=7)

            plt.savefig(anno_file.replace('.npz','_crf_label_filtered_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            #================================
            plt.imshow(img)
            plt.imshow(crf_result_filt_inp-1, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
            plt.axis('off')
            plt.colorbar(shrink=0.5)
            plt.savefig(anno_file.replace('.npz','_image_label_final_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            if save_mode:
                tosave = (np.arange(crf_result_filt_inp.max()) == crf_result_filt_inp[...,None]-1).astype(int)
                savez_dict['final_label'] = tosave.astype('uint8')#crf_result_filt_inp-1
                savez_dict['image'] = (255*img).astype('uint8')
            del img, crf_result_filt_inp

        ### if only one label
        else:
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
                savez_dict['image'] = data['image'].astype('uint8')
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
        orig_distance = 3
    if 'save_mode' not in locals():
        save_mode = True

    print("save mode: %i" % (save_mode))
    print("threshold intra-label distance: %i" % (orig_distance))

    #ok, dooo it
    gen_plot_seq(orig_distance, save_mode)



    # crf_result_filt = filter_one_hot(crf_result, 2*crf_result.shape[0])
    #
    # crf_result_filt = filter_one_hot_spatial(crf_result_filt, 2)
    #
    # plt.subplot(121); plt.imshow(rf_result_filt, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2); plt.axis('off')
    # plt.title('a) Filtered RF output', loc='left', fontsize=7)
    # plt.subplot(122); plt.imshow(crf_result_filt-1, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap); plt.axis('off')
    # plt.title('b) CRF output', loc='left', fontsize=7)
    # plt.savefig(anno_file.replace('.npz','crf_label_filtered_labelgen.png'), dpi=200, bbox_inches='tight')
    # plt.close()
    #
    # plt.imshow(img)
    # plt.imshow(crf_result_filt-1, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
    # plt.axis('off')
    # plt.colorbar(shrink=0.5)
    # plt.savefig(anno_file.replace('.npz','_image_label_CRF_labelgen.png'), dpi=200, bbox_inches='tight')
    # plt.close()


    # # imsave(file.replace('.jpg','_label.png'), final_result)
    # imsave(anno_file.replace('.npz','_label_RF_CRF_col.png'), label_to_colors(final_result, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
    #

    # R = []; W = []
    # counter = 0
    # for k in np.linspace(0,int(img.shape[0]/5),5):
    #     k = int(k)
    #     result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine
    #
    #     #plt.imshow(np.roll(img,k)); plt.imshow(result2, alpha=0.5, cmap=cmap); plt.axis('off'); plt.savefig('CRF_ex_roll'+str(counter)+'.png', dpi=200, bbox_inches='tight'); plt.close()
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
    #     #plt.imshow(np.roll(img,-k)); plt.imshow(result2, alpha=0.5, cmap=cmap); plt.axis('off'); plt.savefig('CRF_ex_roll'+str(counter)+'.png', dpi=200, bbox_inches='tight'); plt.close()
    #
    #     result2 = np.roll(result2, k)
    #     R.append(result2)
    #     counter +=1
    #     if k==0:
    #         W.append(0.1)
    #     else:
    #         W.append(1/np.sqrt(k))
    #
    # #result2 = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')
    # result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
    # del R
    # result = median(result2, disk(DEFAULT_MEDIAN_KERNEL)).astype(np.uint8)-1
    # result[result<0] = 0
    # del result2



# finally:
#     DEFAULT_PEN_WIDTH = 2
#
#     DEFAULT_CRF_DOWNSAMPLE = 2
#
#     DEFAULT_RF_DOWNSAMPLE = 10
#
#     DEFAULT_CRF_THETA = 40
#
#     DEFAULT_CRF_MU = 100
#
#     DEFAULT_MEDIAN_KERNEL = 3
#
#     DEFAULT_RF_NESTIMATORS = 3
#
#     DEFAULT_CRF_GTPROB = 0.9
#
#     SIGMA_MIN = 1
#
#     SIGMA_MAX = 16

    #
    #
    #
    # # use model in predictive mode
    # sh = features.shape
    # features = features.reshape((sh[0], np.prod(sh[1:]))).T
    #
    # label = np.argmax(data['label'],-1).flatten() ##label.flatten()
    #
    # training_data = features[label > 0,:]#.T
    # training_labels = label[label > 0].ravel()
    # del label


    # img = img_to_ubyte_array(file) # read image into memory

    #anno = img_to_ubyte_array(anno_file) # read image into memory
    # label = np.zeros((anno.shape[0], anno.shape[1])).astype('uint8')
    # for counter, c in enumerate(colormap[:-1]):
    #     #print(counter)
    #     #print(c)
    #     mask = (anno[:,:,0]==c[0]) & (anno[:,:,1]==c[1]) & (anno[:,:,0]==c[0]).astype('uint8')
    #     label[mask==1] = counter+1



# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# direc = askdirectory(title='Select directory of corresponding RGB images', initialdir=os.getcwd()+os.sep+'labeled')
# imagefiles = sorted(glob(direc+'/*.jpg'))


# if len(imagefiles)!=len(files):
#     import sys
#     print("The program needs one annotation image per RGB image. Program exiting")
#     sys.exit(2)

# n_estimators = 3

# DEFAULT_CRF_MU = 255
# DEFAULT_CRF_THETA = 10
