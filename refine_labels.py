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
from annotations_to_segmentations import *
from image_segmentation import *

from glob import glob
import skimage.util
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory


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

    return np.array(features)

##========================================================
def img_to_ubyte_array(img):
    """
    PIL.Image.open is used so that a io.BytesIO object containing the image data
    can be passed as img and parsed into an image. Passing a path to an image
    for img will also work.
    """
    try:
       ret = skimage.util.img_as_ubyte(np.array(PIL.Image.open(img)))
    except:
       ret = skimage.util.img_as_ubyte(np.array(PIL.Image.open(img[0])))

    return ret


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
    
    #gt_prob = 0.9
    l_unique = np.unique(label.flatten())#.tolist()
    scale = 1+(5 * (np.array(img.shape).max() / 3000))

    Horig = label.shape[0]
    Worig = label.shape[1]

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

    result = resize(result, (Horig, Worig), order=0, anti_aliasing=True)

    result = rescale(result, orig_mn, orig_mx).astype(np.uint8)

    return result, n

###===========================================================
try:
    from my_defaults import *
    print("Your session defaults loaded")
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



with open('classes.txt') as f:
    classes = f.readlines()

class_label_names = [c.strip() for c in classes]

NUM_LABEL_CLASSES = len(class_label_names)

if NUM_LABEL_CLASSES<=10:
    class_label_colormap = px.colors.qualitative.G10
else:
    class_label_colormap = px.colors.qualitative.Light24


# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

colormap = [
    tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
    for h in [c.replace("#", "") for c in class_label_colormap]
]


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
files = sorted(glob(direc+'/*anno*.png'))


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
direc = askdirectory(title='Select directory of corresponding RGB images', initialdir=os.getcwd()+os.sep+'labeled')
imagefiles = sorted(glob(direc+'/*.jpg'))


if len(imagefiles)!=len(files):
    import sys
    print("The program needs one annotation image per RGB image. Program exiting")
    sys.exit(2)

n_estimators = 3

# DEFAULT_CRF_MU = 255
# DEFAULT_CRF_THETA = 10

for file, anno_file in zip(imagefiles, files):
    print("Working on %s" % (file))
    print("Working on %s" % (anno_file))

    img = img_to_ubyte_array(file) # read image into memory

    anno = img_to_ubyte_array(anno_file) # read image into memory

    label = np.zeros((anno.shape[0], anno.shape[1])).astype('uint8')
    for counter, c in enumerate(colormap[:-1]):
        #print(counter)
        #print(c)
        mask = (anno[:,:,0]==c[0]) & (anno[:,:,1]==c[1]) & (anno[:,:,0]==c[0]).astype('uint8')
        label[mask==1] = counter+1

    features = extract_features(
        img,
        multichannel=True,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    ) # extract image features

    # use model in predictive mode
    sh = features.shape
    features = features.reshape((sh[0], np.prod(sh[1:]))).T

    label = label.flatten()

    training_data = features[label > 0,:]#.T
    training_labels = label[label > 0].ravel()
    del label

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=3)
    clf.fit(training_data, training_labels)

    result = clf.predict(features)
    del features
    result = result.reshape(sh[1:])

    imsave(file.replace('.jpg','_label_RF.png'), result)
    imsave(file.replace('.jpg','_label_RF_col.png'), label_to_colors(result-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

    result = result+1
    # result[:,np.linspace(0,sh[2]-1,100, dtype='int')] = 0
    # result[np.linspace(0,sh[1]-1,100, dtype='int'),:] = 0

    R = []; W = []
    counter = 0
    for k in np.linspace(0,int(img.shape[0]/5),5):
        k = int(k)
        result2, _ = crf_refine(np.roll(result,k), np.roll(img,k), DEFAULT_CRF_THETA, DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE, DEFAULT_CRF_GTPROB) #CRF refine

        #plt.imshow(np.roll(img,k)); plt.imshow(result2, alpha=0.5, cmap=cmap); plt.axis('off'); plt.savefig('CRF_ex_roll'+str(counter)+'.png', dpi=200, bbox_inches='tight'); plt.close()

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

        #plt.imshow(np.roll(img,-k)); plt.imshow(result2, alpha=0.5, cmap=cmap); plt.axis('off'); plt.savefig('CRF_ex_roll'+str(counter)+'.png', dpi=200, bbox_inches='tight'); plt.close()

        result2 = np.roll(result2, k)
        R.append(result2)
        counter +=1
        if k==0:
            W.append(0.1)
        else:
            W.append(1/np.sqrt(k))

    #result2 = np.floor(np.mean(np.dstack(R), axis=-1)).astype('uint8')
    result2 = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
    del R
    result = median(result2, disk(DEFAULT_MEDIAN_KERNEL)).astype(np.uint8)-1
    result[result<0] = 0
    del result2

    imsave(file.replace('.jpg','_label.png'), result)
    imsave(file.replace('.jpg','_label_col.png'), label_to_colors(result-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
