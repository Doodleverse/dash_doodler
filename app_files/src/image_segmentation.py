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

#========================================================
## ``````````````````````````` imports
##========================================================

#numerical
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

#classifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#spatial filters
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy import ndimage
from scipy.signal import convolve2d

#crf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels

#utility
from tempfile import TemporaryFile
from joblib import dump, load, Parallel, delayed
import io, os, logging, psutil, itertools
from skimage.io import imsave
from datetime import datetime
from skimage import filters, feature, img_as_float32
from skimage.transform import resize

#plotly
# import plotly.express as px


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

##====================================
def standardize(img):
    '''
    standardize a 3 band image using adjusted standard deviation
    (1-band images are standardized and returned as 3-band images)
    '''
    #
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    img = rescale(img, 0, 1)
    del m, s, N

    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    return img

##========================================================
def filter_one_hot(label, blobsize):
    #
    '''
    filter the one-hot encoded label images by
    a) converting to a stack of binary one-hote encoded masks
    b) removing small holes and islands
    and
    c) argmax the filtered label stack
    '''
    lstack = (np.arange(label.max()) == label[...,None]-1).astype(int) #one-hot encode

    for kk in range(lstack.shape[-1]):
        l = remove_small_objects(lstack[:,:,kk].astype('uint8')>0, blobsize)
        l = remove_small_holes(lstack[:,:,kk].astype('uint8')>0, blobsize)
        lstack[:,:,kk] = np.round(l).astype(np.uint8)
        del l

    label = np.argmax(lstack, -1)+1
    del lstack
    return label

##========================================================
def filter_one_hot_spatial(label, distance):
    #filter the one-hot encoded  binary masks
    '''
    filter the one-hot encoded label images by
    a) converting to a stack of binary one-hot encoded masks
    b) flagging pixels that are in class transition areas
    c) argmax the filtered label stack
    d) zeroing flagged pixels
    '''
    lstack = (np.arange(label.max()) == label[...,None]-1).astype(int) #one-hot encode

    tmp = np.zeros_like(label)
    for kk in range(lstack.shape[-1]):
        l = lstack[:,:,kk]
        d = ndimage.distance_transform_edt(l)
        l[d<distance] = 0
        lstack[:,:,kk] = np.round(l).astype(np.uint8)
        del l
        tmp[d<=distance] += 1

    label = np.argmax(lstack, -1)+1
    label[tmp==label.max()] = 0
    del lstack
    return label

# ##========================================================
def inpaint_nans(im):
    '''
    quick and dirty nan inpainting using kernel trick
    '''
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = convolve2d((nans==False),ipn_kernel,mode='same',boundary='symm')
        im2 = convolve2d(im,ipn_kernel,mode='same',boundary='symm')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[(nans==False)] = im[(nans==False)]
        im = im2
        nans = np.isnan(im)
    return im

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

    Horig = label.shape[0]
    Worig = label.shape[1]

    l_unique = np.unique(label.flatten())#.tolist()
    scale = 1+(5 * (np.array(img.shape).max() / 3000))
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('CRF scale: %f' % (scale))

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('CRF downsample factor: %f' % (crf_downsample_factor))
    logging.info('CRF theta parameter: %f' % (crf_theta_slider_value))
    logging.info('CRF mu parameter: %f' % (crf_mu_slider_value))
    logging.info('CRF prior probability of labels: %f' % (gt_prob))

    # decimate by factor by taking only every other row and column
    img = img[::crf_downsample_factor,::crf_downsample_factor, :]
    # do the same for the label image
    label = label[::crf_downsample_factor,::crf_downsample_factor]
    # yes, I know this aliases, but considering the task, it is ok; the objective is to
    # make fast inference and resize the output

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Images downsampled by a factor os %f' % (crf_downsample_factor))

    Hnew = label.shape[0]
    Wnew = label.shape[1]

    orig_mn = np.min(np.array(label).flatten())
    orig_mx = np.max(np.array(label).flatten())

    if l_unique[0]==0:
        n = (orig_mx-orig_mn)#+1
    else:

        n = (orig_mx-orig_mn)+1
        label = (label - orig_mn)+1
        mn = np.min(np.array(label).flatten())
        mx = np.max(np.array(label).flatten())

        n = (mx-mn)+1

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
                          schan=(scale,scale,scale),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=crf_mu_slider_value, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC) #260

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('CRF feature extraction complete ... inference starting')

    Q = d.inference(10)
    result = np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8) +1
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('CRF inference made')

    uniq = np.unique(result.flatten())

    result = resize(result, (Horig, Worig), order=0, anti_aliasing=False) #True)

    result = rescale(result, orig_mn, orig_mx).astype(np.uint8)

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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
    gx = filters.gaussian(gx, sigma)
    gy = filters.gaussian(gy, sigma)

    features.append(np.sqrt(gx**2 + gy**2)) #use polar radius of pixel locations as cartesian coordinates

    del gx, gy

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Location features extracted using sigma= %f' % (sigma))

    img_blur = filters.gaussian(img, sigma)

    if intensity:
        features.append(img_blur)

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Intensity features extracted using sigma= %f' % (sigma))

    if edges:
        features.append(filters.sobel(img_blur))

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Edge features extracted using sigma= %f' % (sigma))

    if texture:
        H_elems = [
            np.gradient(np.gradient(img_blur)[ax0], axis=ax1)
            for ax0, ax1 in itertools.combinations_with_replacement(range(img.ndim), 2)
        ]

        eigvals = feature.hessian_matrix_eigvals(H_elems)
        del H_elems

        for eigval_mat in eigvals:
            features.append(eigval_mat)
        del eigval_mat

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Texture features extracted using sigma= %f' % (sigma))

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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

    if (psutil.virtual_memory()[0]>10000000000) & (psutil.virtual_memory()[2]<50): #>10GB and <50% utilization
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Extracting features in parallel')
        logging.info('Total RAM: %i' % (psutil.virtual_memory()[0]))
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        all_results = Parallel(n_jobs=-2, verbose=0)(delayed(features_sigma)(img, sigma, intensity=intensity, edges=edges, texture=texture) for sigma in sigmas)
    else:

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Extracting features in series')
        logging.info('Total RAM: %i' % (psutil.virtual_memory()[0]))
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        n_sigmas = len(sigmas)
        all_results = [
            features_sigma(img, sigma, intensity=intensity, edges=edges, texture=texture)
            for sigma in sigmas
        ]

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Features from channel %i for all scales' % (dim))

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
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Feature extraction complete')

    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))
    logging.info('Memory mapping features to temporary file')

    features = memmap_feats(features)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    return features #np.array(features)

##========================================================
def memmap_feats(features):
    """
    Memory-map data to a temporary file
    """
    features = np.array(features)
    dtype = features.dtype
    feats_shape = features.shape

    outfile = TemporaryFile()
    fp = np.memmap(outfile, dtype=dtype, mode='w+', shape=feats_shape)
    fp[:] = features[:]
    fp.flush()
    del features
    del fp
    logging.info('Features memory mapped features to temporary file: %s' % outfile)

    #read back in again without using any memory
    features = np.memmap(outfile, dtype=dtype, mode='r', shape=feats_shape)
    return features

##========================================================
def do_classify(img,mask,multichannel,intensity,edges,texture,sigma_min,sigma_max, downsample_value):
    """
    Apply classifier to features to extract unary potentials for the CRF
    """
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

    if mask is None:
        raise ValueError("If no classifier clf is passed, you must specify a mask.")
    training_data = features[:, mask > 0].T

    training_data = memmap_feats(training_data)

    training_labels = mask[mask > 0].ravel()

    training_data = training_data[::downsample_value]
    training_labels = training_labels[::downsample_value]

    lim_samples = 100000 #200000

    if training_data.shape[0]>lim_samples:
        logging.info('Number of samples exceeds %i'% lim_samples)
        ind = np.round(np.linspace(0,training_data.shape[0]-1,lim_samples)).astype('int')
        training_data = training_data[ind,:]
        training_labels = training_labels[ind]
        logging.info('Samples have been subsampled')
        logging.info('Number of samples in training data: %i' % (training_data.shape[0]))
        print(training_data.shape)

    clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                solver='adam', alpha=1, random_state=1, max_iter=2000,
                early_stopping=True, hidden_layer_sizes=[100, 60],
            ))
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Initializing MLP model')

    clf.fit(training_data, training_labels)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('MLP model fit to data')

    del training_data, training_labels

    logging.info('Create and memory map model input data')

    data = features[:, mask == 0].T
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    data = memmap_feats(data)
    logging.info('Memory mapped model input data')
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    labels = clf.predict(data)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Model used on data to estimate labels')

    if mask is None:
        result = labels.reshape(img.shape[:2])
        result2 = result.copy()
    else:
        result = np.copy(mask)#+1
        result[mask == 0] = labels
        del labels, mask
        result2 = result.copy()
        del result

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('RF feature extraction and model fitting complete')
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    return result2

##========================================================
def segmentation(
    img,
    img_path,
    results_folder,
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
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
):
    """
    1) Calls do_classify to apply classifier to features to extract unary potentials for the CRF
    then
    2) Calls the spatial filter
    Then
    3) Calls crf_refine to apply CRF
    """

    # #standardization using adjusted standard deviation
    img = standardize(img)

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Image standardized')

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    for ni in np.unique(mask[1:]):
        logging.info('examples provided of %i' % (ni))

    if len(np.unique(mask)[1:])==1:

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Only one class annotation provided, skipping RF and CRF and coding all pixels %i' % (np.unique(mask)[1:]))
        result2 = np.ones(mask.shape[:2])*np.unique(mask)[1:]
        result2 = result2.astype(np.uint8)

    else:

        result = do_classify(img,mask,multichannel,intensity,edges,texture, sigma_min,sigma_max, rf_downsample_value)#,SAVE_RF) # n_estimators,rf_file,data_file,

        Worig = img.shape[0]
        result = filter_one_hot(result, 2*Worig)

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('One-hot labels filtered')

        if Worig>512:
            result = filter_one_hot_spatial(result, 2)

            logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            logging.info('One-hot labels spatially filtered')
        else:
            logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            logging.info('One-hot labels not spatially filtered because width < 512 pixels')

        result = result.astype('float')
        result[result==0] = np.nan
        result = inpaint_nans(result).astype('uint8')

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Spatially filtered values inpainted')

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('RF model applied with sigma range %f : %f' % (sigma_min,sigma_max))
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        def tta_crf_int(img, result, k):
            k = int(k)
            result2, n = crf_refine(np.roll(result,k), np.roll(img,k), crf_theta_slider_value, crf_mu_slider_value, crf_downsample_factor, gt_prob)
            result2 = np.roll(result2, -k)
            if k==0:
                w=.1
            else:
                w = 1/np.sqrt(k)

            return result2, w,n

        num_tta = 5#10

        if (psutil.virtual_memory()[0]>10000000000) & (psutil.virtual_memory()[2]<50): #>10GB and <50% utilization
            logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            logging.info('CRF parallel test-time augmentation')
            logging.info('Total RAM: %i' % (psutil.virtual_memory()[0]))
            logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))
            w = Parallel(n_jobs=-2, verbose=0)(delayed(tta_crf_int)(img, result, k) for k in np.linspace(0,int(img.shape[0])/5,num_tta))
            R,W,n = zip(*w)
        else:
            logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            logging.info('CRF serial test-time augmentation')
            logging.info('Total RAM: %i' % (psutil.virtual_memory()[0]))
            logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))
            R = []; W = []; n = []
            for k in np.linspace(0,int(img.shape[0])/5,num_tta):
                r,w,nn = tta_crf_int(img, result, k)
                R.append(r); W.append(w); n.append(nn)


        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('CRF model applied with %i test-time augmentations' % ( num_tta))

        result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
        del R,W
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Weighted average applied to test-time augmented outputs')

        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('CRF model applied with theta=%f and mu=%f' % ( crf_theta_slider_value, crf_mu_slider_value))
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        if ((n==1)):
            result2[result>0] = np.unique(result)

        result2 = result2.astype('float')
        result2[result2==0] = np.nan
        result2 = inpaint_nans(result2).astype('uint8')
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('Spatially filtered values inpainted')
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    return result2
