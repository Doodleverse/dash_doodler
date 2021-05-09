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


import sys, os, getopt
sys.path.insert(1, '../src')
from annotations_to_segmentations import *
from image_segmentation import *
import matplotlib.pyplot as plt
from glob import glob

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

##====================================
def op1_standardize(img):
    #standardization using adjusted standard deviation
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    img = rescale(img, 0, 1)
    del m, s, N

    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    return img

##====================================
def op3_rf(doodles, features):
    training_data = features[:, doodles > 0].T
    training_labels = doodles[doodles > 0].ravel()
    del doodles

    training_data = training_data[::DEFAULT_RF_DOWNSAMPLE]
    training_labels = training_labels[::DEFAULT_RF_DOWNSAMPLE]

    # scaler = StandardScaler()
    # training_data = scaler.fit_transform(training_data)

    clf = RandomForestClassifier(n_estimators=DEFAULT_RF_NESTIMATORS, n_jobs=-1,class_weight="balanced_subsample", min_samples_split=5)
    clf.fit(training_data, training_labels)
    del training_data, training_labels

    # use model in predictive mode
    sh = features.shape
    features_use = features.reshape((sh[0], np.prod(sh[1:]))).T

    rf_result = clf.predict(features_use)
    del features_use
    rf_result = rf_result.reshape(sh[1:])

    return rf_result

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



##====================================
def op2_features(img):
    if np.ndim(img)==3:
        features = extract_features(
            img,
            multichannel=True,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
        )
    else:
        features = extract_features(
            np.dstack((img,img,img)),
            multichannel=True,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
        )

    return features


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

    return np.array(features)

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

    # all_results = Parallel(n_jobs=-2, verbose=0)(delayed(features_sigma)(img, sigma, intensity=intensity, edges=edges, texture=texture) for sigma in sigmas)

    return list(itertools.chain.from_iterable(all_results))

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
    del gx, gy

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
        del H_elems

        for eigval_mat in eigvals:
            features.append(eigval_mat)
        del eigval_mat

    return features


####================================================
####================================================
####================================================
do_proc = True
orig_distance =2 #3

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

classfile = '/media/marda/TWOTB/USGS/DATA/FloSup/water_masks/dash_doodler-mar15/classes.txt'

with open(classfile) as f:
    classes = f.readlines()
class_string = '_'.join([c.strip() for c in classes])

class_label_colormap = px.colors.qualitative.G10
NCLASSES = len(classes) #4

uniqs = [
    tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
    for h in [c.replace("#", "") for c in class_label_colormap]
][:NCLASSES]


# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# imagedirec = askdirectory(title='Select directory of images', initialdir=os.getcwd()+os.sep+'results')
# imagedirec = '/media/marda/TWOTB/USGS/DATA/FloSup/water_masks/dash_doodler-mar15/labeled'

imagedirec = '/media/marda/TWOTB/USGS/DATA/FloSup/water_masks/oblique-binary/images/images'

# get the assocuated images
imagefiles = sorted(glob(imagedirec+os.sep+'*.jpg'))

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
# direc = '/media/marda/TWOTB/USGS/DATA/FloSup/water_masks/dash_doodler-mar15/results/results2021-02-23-20-02'

direc = '/media/marda/TWOTB/USGS/DATA/FloSup/water_masks/oblique-binary/dash_doodler/results'

direcs = [x[0] for x in os.walk(direc)]

for direc in direcs[1:]:

    files = sorted(glob(direc+'/*anno*.png'))

    # get only the last set of annotataions per image
    roots = [f.split(os.sep)[-1].split('.png')[0] for f in files]

    rootroots = [r.split('annotations')[0] for r in roots]

    G = []
    for r in np.unique(rootroots):
        g = [f for f in files if r in f]
        G.append(g[-1])

    use_files = [ sorted(g)[-1] for g in G]

    use_imagefiles = []
    for u in use_files:
        for i in imagefiles:
            if u.split(os.sep)[-1].split('_annotations')[0] in i:
                use_imagefiles.append(i)

    #### loop through each file
    for file, image_file in zip(use_files,use_imagefiles):
        print("Working on %s" % (file))

        data = dict()
        data['settings'] = np.array([ 3. ,  4. , 10. ,  1. ,  1. ,  3. ,  0.9,  1. , 16. ])

        for value,name in zip( data['settings'],['DEFAULT_PEN_WIDTH', 'DEFAULT_CRF_DOWNSAMPLE',\
                'DEFAULT_RF_DOWNSAMPLE', 'DEFAULT_CRF_THETA', 'DEFAULT_CRF_MU', \
                'DEFAULT_RF_NESTIMATORS', 'DEFAULT_CRF_GTPROB', 'SIGMA_MIN', 'SIGMA_MAX']):
                if name is not 'DEFAULT_CRF_GTPROB':
                    exec(name+'='+str(int(value)))
                else:
                    exec(name+'='+str(value))

        img = img_to_ubyte_array(image_file) # read image into memory
        # computations are faster as float32
        img = img_as_float32(img)
        img = op1_standardize(img)

        features = op2_features(img)

        label = img_to_ubyte_array(file) # read image into memory

        ##find unique RGB combos in label, MxNx3 color matrix
        #uniqs = np.unique(label.reshape(-1,3),axis=0)
        ##pre-allocate one-hot encoded label stack, MxNxC
        ##(c=number of classes or unique rgb values)
        lstack = np.zeros( (label.shape[0], label.shape[1], 1+len(uniqs)) )
        ##enumerate rgb tuples
        for counter,rgb in enumerate(uniqs):
            ##populate layers of lstack with ones where label matches the rgb tuple
            lstack[:,:,1+counter] = np.sum( (label==tuple(rgb)).astype('int'), axis=-1) #[:,:,0]

        del label
        ## find the backgorund element (assumes one with most ones)
        # ind_del = np.argmax([np.sum(l) for l in lstack.T])
        # ## deletes that layer from the stack
        # lstack = np.delete(lstack, ind_del, 2)


        data['doodles'] = np.argmax(lstack,-1)
        del lstack

        rf_result = op3_rf(data['doodles'], features)
        del features
        rf_result_filt = filter_one_hot(rf_result, 2*rf_result.shape[0])
        rf_result_filt = filter_one_hot_spatial(rf_result_filt, orig_distance)
        rf_result_filt = rf_result_filt.astype('float')
        rf_result_filt[rf_result_filt==0] = np.nan
        rf_result_filt_inp = inpaint_nans(rf_result_filt).astype('uint8')
        del rf_result_filt, rf_result

        w = Parallel(n_jobs=-2, verbose=0)(delayed(tta_crf)(img, rf_result_filt_inp, k) for k in np.linspace(0,int(img.shape[0]),10))
        R,W,n = zip(*w)
        del rf_result_filt_inp
        crf_result = np.round(np.average(np.dstack(R), axis=-1, weights = W)).astype('uint8')
        del R,W,n
        crf_result_filt = filter_one_hot(crf_result, 2*crf_result.shape[0])
        crf_result_filt = filter_one_hot_spatial(crf_result_filt, orig_distance)
        crf_result_filt = crf_result_filt.astype('float')
        crf_result_filt[crf_result_filt==0] = np.nan
        crf_result_filt_inp = inpaint_nans(crf_result_filt).astype('uint8')
        del crf_result_filt, crf_result

        lstack = (np.arange(crf_result_filt_inp.max()) == crf_result_filt_inp[...,None]-1).astype(int) #one-hot encode
        data['label'] = lstack

        # plt.subplot(121); plt.imshow(img); plt.imshow(data['doodles'], alpha=0.5);
        # plt.subplot(122); plt.imshow(img); plt.imshow(crf_result_filt_inp, alpha=0.5);
        # plt.show(); plt.close()
        del crf_result_filt_inp, lstack

        data['color_doodles'] = np.zeros( (data['doodles'].shape[0], data['doodles'].shape[1], 3) , dtype='uint8')
        ##enumerate rgb tuples
        for counter,rgb in enumerate(uniqs):
            data['color_doodles'][:,:,0][data['doodles']==counter+1] = rgb[0]
            data['color_doodles'][:,:,1][data['doodles']==counter+1] = rgb[1]
            data['color_doodles'][:,:,2][data['doodles']==counter+1] = rgb[2]

        del data['doodles']
        data['image'] = img
        np.savez(file.replace('.png','_'+class_string+'.npz'), **data )
        del data, img
