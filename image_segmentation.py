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
from skimage.io import imsave#, imread

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk

np.seterr(divide='ignore', invalid='ignore')

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def expand_img(img):
    '''
    expands a 3-band image into a 6-band image stack,
    with the last three bands being derived from the first 3
    specifically; 1) VARI = (G-R)/(G+R-B); 2) NEXG = (2*G - R - B) / (G+R+B); 3) NGRDI = (G-R)/(G+R)
    '''
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    VARI = 1+(G-R)/1+(G+R-B)
    NEXG = 1+(2*G - R - B) / 1+(G+R+B)
    NGRDI = 1+(G-R)/1+(G+R)
    VARI[np.isinf(VARI)] = 1e-2
    NEXG[np.isinf(NEXG)] = 1e-2
    NGRDI[np.isinf(NGRDI)] = 1e-2
    VARI[np.isnan(VARI)] = 1e-2
    NEXG[np.isnan(NEXG)] = 1e-2
    NGRDI[np.isnan(NGRDI)] = 1e-2
    VARI[VARI==0] = 1e-2
    NEXG[NEXG==0] = 1e-2
    NGRDI[NGRDI==0] = 1e-2

    VARI = rescale(np.log(VARI),0,255)
    NEXG = rescale(np.log(NEXG),0,255)
    NGRDI = rescale(np.log(NGRDI),0,255)

    STACK = np.dstack((R,G,B,VARI,NEXG,NGRDI)).astype(np.int)
    del R, G, B, VARI, NEXG, NGRDI
    return STACK

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
    img):
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

    mn = np.min(np.array(label).flatten())
    mx = np.max(np.array(label).flatten())

    label = rescale(np.array(label),mn,mx).astype(np.int)

    H = label.shape[0]
    W = label.shape[1]
    U = unary_from_labels(label,1+(mx-mn),gt_prob=0.66)
    d = dcrf.DenseCRF2D(H, W, 1+(mx-mn)) #5
    d.setUnaryEnergy(U)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(13, 13),
                 compat=130,
                 kernel=dcrf.DIAG_KERNEL,
                 normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(
                          sdims=(60, 60),
                          schan=(2,2,2,2,2,2),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=260,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    result = np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8)
    result = rescale(result, mn, mx).astype(np.uint8)

    print("CRF post-processing complete")
    return result

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

    return features

##========================================================
def extract_features_2d(
    img,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16
):
    """Features for a single channel image. ``img`` can be 2d or 3d.
    """
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
    return np.array(features)


##========================================================
def label_to_colors(
    img,
    colormap=px.colors.qualitative.G10,
    color_class_offset=0
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    return cimg

##========================================================
def segmentation(
    img,
    img_path,
    median_filter_value,
    mask=None,
    multichannel=True,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
    downsample=10,
    clf=None,
):

    """
    Segmentation using labeled parts of the image and a random forest classifier.
    """
    # t1 = time()
    features = extract_features(
        img,
        multichannel=multichannel,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    # t2 = time()
    if clf is None:
        if mask is None:
            raise ValueError("If no classifier clf is passed, you must specify a mask.")
        training_data = features[:, mask > 0].T
        training_labels = mask[mask > 0].ravel()
        data = features[:, mask == 0].T
        # t3 = time()
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(training_data[::downsample], training_labels[::downsample])
        result = np.copy(mask)

    else:
        # we have to flatten all but the first dimension of features
        data = features.reshape((features.shape[0], np.product(features.shape[1:]))).T
        # t3 = time()
    # t4 = time()

    labels = clf.predict(data)
    if mask is None:
        result = labels.reshape(img.shape[:2])
    else:
        result[mask == 0] = labels

    print("applying CRF refinement:")
    result = crf_refine(result, expand_img(img))

    if "Apply Median Filter" in median_filter_value:
        print("applying median filter:")
        result = median(result, disk(20)).astype(np.uint8)


    if type(img_path) is list:
        imsave(img_path[0].replace('assets/','results/').replace('.jpg','_label.png'), label_to_colors(result-1)) #result)
    else:
        imsave(img_path.replace('assets/','results/').replace('.jpg','_label.png'), label_to_colors(result-1)) #result)

    if type(img_path) is list:
        imsave(img_path[0].replace('assets/','results/').replace('.jpg','_label_greyscale.png'), result)
    else:
        imsave(img_path.replace('assets/','results/').replace('.jpg','_label_greyscale.png'), result)

    return result, clf
