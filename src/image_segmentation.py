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
from joblib import dump, load
import io, os, logging

np.seterr(divide='ignore', invalid='ignore')

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

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
    Q = d.inference(10)
    result = 1+np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF inference made')

    result = resize(result, (Horig, Worig), order=0, anti_aliasing=True)

    result = rescale(result, orig_mn, orig_mx).astype(np.uint8)

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('CRF post-processing complete')

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

    n_sigmas = len(sigmas)
    all_results = [
        features_sigma(img, sigma, intensity=intensity, edges=edges, texture=texture)
        for sigma in sigmas
    ]
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
        features = extract_features_2d(
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
def do_rf(img,rf_file,mask,multichannel,intensity,edges,texture,sigma_min,sigma_max, downsample_value, n_estimators):

    features = extract_features(
        img,
        multichannel=multichannel,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    if downsample_value is None:
        downsample_value = 10

    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Using %i decision trees per image' % (n_estimators))

    if mask is None:
        raise ValueError("If no classifier clf is passed, you must specify a mask.")
    training_data = features[:, mask > 0].T
    training_labels = mask[mask > 0].ravel()
    data = features[:, mask == 0].T
    try:
        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Updating existing RF classifier')
        #rf_file = 'RandomForestClassifier.pkl.z'

        clf = load(rf_file) #load last model from file
        clf.n_estimators += n_estimators #add more trees for the new data
        clf.fit(training_data[::downsample_value], training_labels[::downsample_value]) # fit with with new data
        os.remove(rf_file) #remove old file
        dump(clf, rf_file, compress=True) #save new file
    except:
        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Initialize RF classifier with %i estimators' % (n_estimators))
        ##warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1) #, warm_start=True)
        clf.fit(training_data[::downsample_value], training_labels[::downsample_value])
        dump(clf, rf_file, compress=True)

    result = np.copy(mask)#+1

    labels = clf.predict(data)
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
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
    median_filter_value,
    rf_downsample_value,
    crf_downsample_factor,
    gt_prob,
    mask=None,
    multichannel=True,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
    n_estimators=5
):

    if 'rf' in callback_context:
        if 'crf' not in callback_context:

            result2 = do_rf(img,rf_file,mask,multichannel,intensity,edges,texture,sigma_min,sigma_max, rf_downsample_value, n_estimators)
            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('RF model applied with sigma range %f : %f' % (sigma_min,sigma_max))

            # if 'median' in callback_context:
            if median_filter_value>1: #"Apply Median Filter" in median_filter_value:
                result2 = median(result2, disk(median_filter_value)).astype(np.uint8)
                logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                logging.info('Median filter of radius %i applied' % (median_filter_value))

    if 'crf' in callback_context:
        if crf_theta_slider_value is None:
            result2 = result.copy()
        else:

            result = do_rf(img,rf_file,mask,True,True,False,False,sigma_min,sigma_max, rf_downsample_value, n_estimators)
            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('RF model applied with sigma range %f : %f' % (sigma_min,sigma_max))

            result2, n = crf_refine(result, img, crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob) #result
            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('CRF model applied with theta=%f and mu=%f' % ( crf_theta_slider_value, crf_mu_slider_value))

            if ((n==1)):
                result2[result>0] = np.unique(result)

        if median_filter_value>1: #"Apply Median Filter" in median_filter_value:
            result2 = median(result2, disk(median_filter_value)).astype(np.uint8)

            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('Median filter of radius %i applied' % (median_filter_value))
    return result2
