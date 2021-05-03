# Written by Dr Daniel Buscombe, Marda Science LLC
# for "ML Mondays", a course supported by the USGS Community for Data Integration
# and the USGS Coastal Change Hazards Program
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

import itertools
import numpy as np
from skimage import filters, feature, img_as_float32
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from skimage.io import imsave
from datetime import datetime

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.transform import resize
from joblib import dump, load, Parallel, delayed
import io, os, logging

np.seterr(divide='ignore', invalid='ignore')

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def rescale(dat,
    mn,
    mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn

##========================================================
def crf_refine(label,
    img,
    crf_theta_slider_value,
    crf_mu_slider_value,
    crf_downsample_factor,
    gt_prob):
    """
    "crf_refine(label, img)"
    This function refines a label image based on an input label image and the associated image
    Uses a conditional random field algorithm using spatial and image features
    INPUTS:
        * label [ndarray]: label image 2D matrix of integers
        * image [ndarray]: image 3D matrix of integers
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: label [ndarray]: label image 2D matrix of integers
    """

    gx,gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    # print(gx.shape)
    img = np.dstack((img,np.sqrt(gx**2 + gy**2))) #gx,gy))
    #print(img.shape)

    #gt_prob = 0.9
    l_unique = np.unique(label.flatten())#.tolist()
    scale = 1+(5 * (np.array(img.shape).max() / 3000))
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF scale: %f' % (scale))

    Horig = label.shape[0]
    Worig = label.shape[1]

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF downsample factor: %f' % (crf_downsample_factor))
    logging.info('CRF theta parameter: %f' % (crf_theta_slider_value))
    logging.info('CRF mu parameter: %f' % (crf_mu_slider_value))
    logging.info('CRF prior probability of labels: %f' % (gt_prob))

    # decimate by factor by taking only every other row and column
    img = img[::crf_downsample_factor,::crf_downsample_factor, :]
    # do the same for the label image
    label = label[::crf_downsample_factor,::crf_downsample_factor]

    orig_mn = np.min(np.array(label).flatten())
    orig_mx = np.max(np.array(label).flatten())

    n = 1+(orig_mx-orig_mn)

    label = 1+(label - orig_mn)

    mn = np.min(np.array(label).flatten())
    mx = np.max(np.array(label).flatten())

    n = 1+(mx-mn)

    H = label.shape[0]
    W = label.shape[1]
    U = unary_from_labels(label.astype('int'), n, gt_prob=gt_prob)
    d = dcrf.DenseCRF2D(H, W, n)
    d.setUnaryEnergy(U)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(3, 3),
                 compat=3,
                 kernel=dcrf.DIAG_KERNEL,
                 normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(
                          sdims=(crf_theta_slider_value, crf_theta_slider_value),
                          # schan=(2,2,2,2,2,2), #add these when implement 6 band
                          schan=(scale,scale,scale),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=crf_mu_slider_value, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC) #260

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF feature extraction complete ... inference starting')

    Q = d.inference(10)
    result = 1+np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF inference made')

    result = resize(result, (Horig, Worig), order=0, anti_aliasing=True)

    result = rescale(result, orig_mn, orig_mx).astype(np.uint8)

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('label resized and rescaled ... CRF post-processing complete')

    return result, n


##========================================================
def features_sigma(img,
    sigma,
    intensity=True,
    edges=True,
    texture=True):
    """Features for a single value of the Gaussian blurring parameter ``sigma``
    """

    features = []

    gx,gy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    # print(gx.shape)
    #features.append(gx)
    features.append(np.sqrt(gx**2 + gy**2)) #gy) #use polar radius of pixel locations as cartesian coordinates
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Location features extracted')

    img_blur = filters.gaussian(img, sigma)

    if intensity:
        features.append(img_blur)

    if edges:
        features.append(filters.sobel(img_blur))

    if texture:
        H_elems = [
            np.gradient(np.gradient(img_blur)[ax0], axis=ax1)
            for ax0, ax1 in itertools.combinations_with_replacement(range(img.ndim), 2)
        ]

        eigvals = feature.hessian_matrix_eigvals(H_elems)

        for eigval_mat in eigvals:
            features.append(eigval_mat)

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Image features extracted using sigma= %f' % (sigma))

    return features

##========================================================
def extract_features_2d(
    dim,
    img,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16
):
    """Features for a single channel image. ``img`` can be 2d or 3d.
    """

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Extracting features from channel %i' % (dim))

    # computations are faster as float32
    img = img_as_float32(img)

    sigmas = np.logspace(
        np.log2(sigma_min),
        np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2,
        endpoint=True,
    )

    #n_sigmas = len(sigmas)
    all_results = [
        features_sigma(img, sigma, intensity=intensity, edges=edges, texture=texture)
        for sigma in sigmas
    ]

    all_results = Parallel(n_jobs=-1, verbose=0)(delayed(features_sigma)(img, sigma, intensity=intensity, edges=edges, texture=texture) for sigma in sigmas)

    return list(itertools.chain.from_iterable(all_results))

##========================================================
def extract_features(
    img,
    multichannel=True,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
):
    """Features for a single- or multi-channel image.
    """
    if multichannel: #img.ndim == 3 and multichannel:
        all_results = (
            extract_features_2d(
                dim,
                img[..., dim],
                intensity=intensity,
                edges=edges,
                texture=texture,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            for dim in range(img.shape[-1])
        )
        features = list(itertools.chain.from_iterable(all_results))
    else:
        features = extract_features_2d(0,
            img,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Feature extraction complete')

    return np.array(features)


##========================================================
def do_rf(img,rf_file,data_file,mask,multichannel,intensity,edges,texture,sigma_min,sigma_max, downsample_value, n_estimators):

    if np.ndim(img)==3:
        features = extract_features(
            img,
            multichannel=multichannel,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        features = extract_features(
            np.dstack((img,img,img)),
            multichannel=multichannel,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Using %i decision trees per image' % (n_estimators))

    if mask is None:
        raise ValueError("If no classifier clf is passed, you must specify a mask.")
    training_data = features[:, mask > 0].T
    training_labels = mask[mask > 0].ravel()
    data = features[:, mask == 0].T
    # try:
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Updating existing RF classifier')

    training_data = training_data[::downsample_value]
    training_labels = training_labels[::downsample_value]

    try:
        file_training_data, file_training_labels = load(data_file)

        training_data = np.concatenate((file_training_data, training_data))
        training_labels = np.concatenate((file_training_labels, training_labels))
        logging.info('Samples concatenated with those from file')
        logging.info('Number of samples in training data: %i' % (training_data.shape[0]))

    except:
        pass

    #print(training_data.shape)
    #print(training_labels.shape)

    if training_data.shape[0]>500000:
        logging.info('Number of samples exceeds 500000')
        ind = np.round(np.linspace(0,training_data.shape[0]-1,500000)).astype('int')
        training_data = training_data[ind,:]
        training_labels = training_labels[ind]
        logging.info('Samples have been subsampled')
        logging.info('Number of samples in training data: %i' % (training_data.shape[0]))
        print(training_data.shape)

    try:
        clf = load(rf_file) #load last model from file
        # path = clf.cost_complexity_pruning_path(training_data, training_labels)

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Loading model from %s' % (rf_file))
        logging.info('Number of trees: %i' % (clf.n_estimators))
    except:
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)#, ccp_alpha=0.02)
        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Initializing RF model')
    clf.n_estimators += n_estimators #add more trees for the new data
    #print(clf.n_estimators)
    clf.fit(training_data, training_labels) # fit with with new data
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('RF model fit to data')

    dump(clf, rf_file, compress=True) #save new file
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Model saved to %s'% rf_file)
    dump((training_data, training_labels), data_file, compress=True) #save new file
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Data saved to %s'% data_file)

    result = np.copy(mask)#+1

    labels = clf.predict(data)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Model used on data to estimate labels')

    if mask is None:
        result = labels.reshape(img.shape[:2])
        result2 = result.copy()
    else:
        result[mask == 0] = labels
        result2 = result.copy()

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('RF feature extraction and model fitting complete')

    return result2


##========================================================
def segmentation(
    img,
    img_path,
    results_folder,
    rf_file,
    data_file,
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
    median_filter_value,
    rf_downsample_value,
    crf_downsample_factor,
    gt_prob,
    mask,#=None,
    multichannel,#=True,
    intensity,#=True,
    edges,#=True,
    texture,#=True,
    sigma_min,#=0.5,
    sigma_max,#=16,
    n_estimators,#=5
):

    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    N = np.prod(np.shape(img))
    s = np.maximum(np.sqrt(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    for ni in np.unique(mask[1:]):
        logging.info('examples provided of %i' % (ni))

    if len(np.unique(mask)[1:])==1:

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Only one class annotation provided, skipping RF and CRF and coding all pixels %i' % (np.unique(mask)[1:]))
        result2 = np.ones(mask.shape[:2])*np.unique(mask)[1:]
        result2 = result2.astype(np.uint8)

    else:

        result = do_rf(img,rf_file,data_file,mask,multichannel,intensity,edges,texture, sigma_min,sigma_max, rf_downsample_value, n_estimators) #

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('RF model applied with sigma range %f : %f' % (sigma_min,sigma_max))

        # result2 = result.copy()

        # R = []; W = []
        # counter = 0
        # for k in np.linspace(0,int(img.shape[0]/5),5):
        #     k = int(k)
        #     result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob) #CRF refine
        #     result2 = np.roll(result2, -k)
        #     R.append(result2)
        #     logging.info('CRF model applied with roll of %i' % (k))
        #     counter +=1
        #     if k==0:
        #         W.append(.1) #np.nan)
        #     else:
        #         W.append(1/np.sqrt(k))

        # for k in np.linspace(0,int(img.shape[0]/5),5):
        #     k = int(k)
        #     result2, n = crf_refine(np.roll(result,-k), np.roll(img,-k), crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob) #CRF refine
        #     result2 = np.roll(result2, k)
        #     R.append(result2)
        #     logging.info('CRF model applied with roll of -%i' % (k))
        #     counter +=1
        #     if k==0:
        #         W.append(.1) #np.nan)
        #     else:
        #         W.append(1/np.sqrt(k))

        def tta_crf(img, result, k):
            k = int(k)
            result2, n = crf_refine(np.roll(result,k), np.roll(img,k), crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob) #CRF refine
            result2 = np.roll(result2, -k)
            if k==0:
                w=.1
            else:
                w = 1/np.sqrt(k)
            return result2, w,n

        w = Parallel(n_jobs=-2, verbose=0)(delayed(tta_crf)(img, result, k) for k in np.linspace(0,int(img.shape[0]/5),5))
        R,W,n = zip(*w)

        result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
        del R

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('CRF model applied with theta=%f and mu=%f' % ( crf_theta_slider_value, crf_mu_slider_value))
        #
        # result2, n = crf_refine(result, img, crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob) #CRF refine

        if ((n==1)):
            result2[result>0] = np.unique(result)

        if median_filter_value>1: #"Apply Median Filter" in median_filter_value:
            result2 = median(result2, disk(median_filter_value)).astype(np.uint8)

            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('Median filter of radius %i applied' % (median_filter_value))
    return result2


# ##========================================================
# def expand_img(img):
#     '''
#     expands a 3-band image into a 6-band image stack,
#     with the last three bands being derived from the first 3
#     specifically; 1) VARI = (G-R)/(G+R-B); 2) NEXG = (2*G - R - B) / (G+R+B); 3) NGRDI = (G-R)/(G+R)
#     '''
#     R = img[:,:,0]
#     G = img[:,:,1]
#     B = img[:,:,2]
#
#     VARI = 1+(G-R)/1+(G+R-B)
#     NEXG = 1+(2*G - R - B) / 1+(G+R+B)
#     NGRDI = 1+(G-R)/1+(G+R)
#     VARI[np.isinf(VARI)] = 1e-2
#     NEXG[np.isinf(NEXG)] = 1e-2
#     NGRDI[np.isinf(NGRDI)] = 1e-2
#     VARI[np.isnan(VARI)] = 1e-2
#     NEXG[np.isnan(NEXG)] = 1e-2
#     NGRDI[np.isnan(NGRDI)] = 1e-2
#     VARI[VARI==0] = 1e-2
#     NEXG[NEXG==0] = 1e-2
#     NGRDI[NGRDI==0] = 1e-2
#
#     VARI = rescale(np.log(VARI),0,255)
#     NEXG = rescale(np.log(NEXG),0,255)
#     NGRDI = rescale(np.log(NGRDI),0,255)
#
#     STACK = np.dstack((R,G,B,VARI,NEXG,NGRDI)).astype(np.int)
#     del R, G, B, VARI, NEXG, NGRDI
#     return STACK

    # except:
    #     logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    #     logging.info('Initialize RF classifier with %i estimators' % (n_estimators))
    #     ##warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    #     clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1) #, warm_start=True)
    #     clf.fit(training_data[::downsample_value], training_labels[::downsample_value])
    #     dump(clf, rf_file, compress=True)
    # if 'result2' not in locals(): #''crf' in callback_context:
    #     # if crf_theta_slider_value is None:
    #     result2 = result.copy()
    # else:

    #result = do_rf(img,rf_file,mask,True,True,False,False,sigma_min,sigma_max, rf_downsample_value, n_estimators) #multichannel,intensity,edges,texture,
