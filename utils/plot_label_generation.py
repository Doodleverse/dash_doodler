# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-2022, Marda Science LLC
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
import sys, os, getopt, shutil
sys.path.insert(1, '../')
from doodler_engine.annotations_to_segmentations import *
from doodler_engine.image_segmentation import *


from glob import glob
from tqdm import tqdm

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

import matplotlib
import matplotlib.pyplot as plt

from imageio import imwrite
import plotly.express as px

###===========================================================
try:
    sys.path.insert(1, '../')
    from my_defaults import *
    print("Your session defaults loaded")
except:
    DEFAULT_PEN_WIDTH = 3
    DEFAULT_CRF_DOWNSAMPLE = 1
    DEFAULT_RF_DOWNSAMPLE = 1
    DEFAULT_CRF_THETA = 1
    DEFAULT_CRF_MU = 1
    DEFAULT_CRF_GTPROB = 0.9
    DEFAULT_NUMSCALES = 3

###===========================================================
def gen_plot_seq():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')
    files = sorted(glob(direc+'/*.npz'))

    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]

    #### loop through each file
    for anno_file in tqdm(files):

        overdir = os.path.join(direc, 'label_generation')

        try:
            os.mkdir(overdir)
        except:
            pass

        print("Working on %s" % (anno_file))
        dat = np.load(anno_file)
        data = dict()
        for k in dat.keys():
            data[k] = dat[k]
        del dat

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

            if 'orig_image' in data.keys():
                img = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
            else:
                img = np.squeeze(data['image'].astype('uint8'))[:,:,:3]
                data['orig_image'] = data['image']

            #================================
            ##fig1 - img versus standardized image
            plt.subplot(121)
            plt.imshow(img); plt.axis('off')
            plt.title('a) Original image', loc='left', fontsize=7)

            # #standardization using adjusted standard deviation
            img = standardize(img)

            #================================
            ##fig2 - img / doodles
            plt.subplot(122)
            plt.imshow(img); plt.axis('off')
            plt.title('b) Standardized image', loc='left', fontsize=7)
            plt.savefig(anno_file.replace('.npz','_image_filt_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            tmp = data['doodles'].astype('float')
            tmp[tmp==0] = np.nan

            ## do plot of images and doodles
            plt.imshow(img[:,:,0], cmap='gray')
            plt.imshow(tmp, alpha=0.5, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap2) 
            plt.axis('off')
            plt.colorbar(shrink=0.5)
            plt.title('Doodles on grayscale image', loc='left', fontsize=7)            
            plt.savefig(anno_file.replace('.npz','_image_doodles_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()
            del tmp

            #####=========================== MLP

            if np.ndim(img)==3:
                features = extract_features(
                    img,
                    DEFAULT_NUMSCALES,
                    multichannel=True,
                    intensity=True,
                    edges=True,
                    texture=True,
                    sigma_min=0.5, #SIGMA_MIN,
                    sigma_max=16, #SIGMA_MAX,
                )
            else:
                features = extract_features(
                    np.dstack((img,img,img)),
                    DEFAULT_NUMSCALES,
                    multichannel=True,
                    intensity=True,
                    edges=True,
                    texture=True,
                    sigma_min=0.5, #SIGMA_MIN,
                    sigma_max=16, #SIGMA_MAX,
                )

            counter=1
            for k in [0,1,2,3,4]:
                plt.subplot(2,5,counter)
                plt.imshow(features[k].reshape((img.shape[0], img.shape[1])), cmap='gray'); plt.axis('off')
                if k==0:
                    plt.title('a) Smallest scale', loc='left', fontsize=7)
                counter+=1

            for k in np.arange(features.shape[0]-5,features.shape[0],1):
                k = np.int(k)
                plt.subplot(2,5,counter)
                plt.imshow(features[k].reshape((img.shape[0], img.shape[1])), cmap='gray'); plt.axis('off')
                if counter==10:
                    plt.subplot(2,5,6)
                    plt.title('b) Largest scale', loc='left', fontsize=7)
                counter+=1

            plt.savefig(anno_file.replace('.npz','_image_feats_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            #================================
            # MLP analysis
            n=len(np.unique(data['doodles'])[1:])

            mlp_result, unique_labels = do_classify(data['orig_image'],data['doodles'], n, #DEFAULT_NUMSCALES
                                        True,True,True,True,
                                        0.5,16, DEFAULT_RF_DOWNSAMPLE)
            
            n=len(unique_labels)
            mlp_result = mlp_result.reshape(data['orig_image'].shape[0],data['orig_image'].shape[1],len(unique_labels))
            savez_dict['mlp_result_softmax'] = mlp_result

            mlp_result = np.argmax(mlp_result,-1)+1

            uniq_doodles = np.unique(data['doodles'])[1:]
            uniq_mlp = np.unique(mlp_result)
            mlp_result2 = np.zeros_like(mlp_result)
            for o,e in zip(uniq_doodles,uniq_mlp):
                mlp_result2[mlp_result==e] = o

            mlp_result = mlp_result2.copy()-1

            #================================
            plt.imshow(data['orig_image'])
            plt.imshow(mlp_result, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
            plt.axis('off')
            plt.colorbar(shrink=0.5)
            plt.savefig(anno_file.replace('.npz','_image_label_MLP_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            savez_dict['mlp_result_label'] = mlp_result ##np.argmax(,-1)

            # make a limited one-hot array and add the available bands
            nx, ny = mlp_result.shape
            mlp_result_softmax = np.zeros((nx,ny,n)) #NUM_LABEL_CLASSES
            mlp_result_softmax[:,:,:n] = (np.arange(n) == 1+mlp_result[...,None]-1).astype(int) #NUM_LABEL_CLASSES

            if not n==len(np.unique(np.argmax(mlp_result_softmax,-1))):
            # if not np.all(uniq_doodles-1==np.unique(np.argmax(mlp_result_softmax,-1))):
                print("MLP failed")

                try:
                    crf_result, n = crf_refine_from_integer_labels(data['doodles'], data['orig_image'], n, #NUM_LABEL_CLASSES,
                                                                DEFAULT_CRF_THETA, DEFAULT_CRF_MU, 
                                                                DEFAULT_CRF_DOWNSAMPLE)

                    uniq_crf = np.unique(crf_result)
                    crf_result2 = np.zeros_like(crf_result)
                    for o,e in zip(uniq_doodles,uniq_crf):
                        crf_result2[crf_result==e] = o

                    crf_result = crf_result2.copy()-1
                except:
                    crf_result = mlp_result.copy()
            else:
                #================================
                # CRF analysis           
                print('CRF ...')

                try:
                    crf_result, n = crf_refine(mlp_result_softmax, data['orig_image'], n, #NUM_LABEL_CLASSES,
                                            DEFAULT_CRF_THETA, 
                                            DEFAULT_CRF_MU, DEFAULT_CRF_DOWNSAMPLE)

                    uniq_crf = np.unique(crf_result)
                    crf_result2 = np.zeros_like(crf_result)
                    for o,e in zip(uniq_doodles,uniq_crf):
                        crf_result2[crf_result==e] = o

                    crf_result = crf_result2.copy()-1

                    if not len(uniq_doodles)==len(np.unique(crf_result)):
                    # if not np.all(uniq_doodles-1==np.unique(crf_result)):
                        print("CRF failed")

                        crf_result, n = crf_refine_from_integer_labels(data['doodles'], data['orig_image'], n, #NUM_LABEL_CLASSES,
                                                                    DEFAULT_CRF_THETA, DEFAULT_CRF_MU, 
                                                                    DEFAULT_CRF_DOWNSAMPLE)

                        uniq_crf = np.unique(crf_result)
                        crf_result2 = np.zeros_like(crf_result)
                        for o,e in zip(uniq_doodles,uniq_crf):
                            crf_result2[crf_result==e] = o

                        crf_result = crf_result2.copy()-1

                except:
                    crf_result = mlp_result.copy()

            #================================
            plt.imshow(data['orig_image'])
            plt.imshow(crf_result, alpha=0.25, vmin=0, vmax=NUM_LABEL_CLASSES, cmap=cmap) #'inferno')
            plt.axis('off')
            plt.colorbar(shrink=0.5)
            plt.savefig(anno_file.replace('.npz','_image_label_CRF_labelgen.png'), dpi=200, bbox_inches='tight')
            plt.close()

            # crf_result = crf_result-1

            # match = np.unique(np.argmax(mlp_result,-1))
            # match2 = np.unique(crf_result)
            # # print(match2)
            # if not np.all(np.array(match)==np.array(match2)):
            #     print("Problem with CRF solution.... reverting back to MLP solution")
            #     crf_result = np.argmax(mlp_result,-1).copy()


            tosave = (np.arange(crf_result.max()) == crf_result[...,None]-1).astype(int)
            savez_dict['final_label'] = tosave.astype('uint8')#crf_result_filt_inp-1
            savez_dict['image'] = data['orig_image'] #(255*img).astype('uint8')
            #del img, crf_result

            imwrite(anno_file.replace('.npz','_label_labelgen.png'), np.argmax(savez_dict['final_label'],-1).astype('uint8'))
            imwrite(anno_file.replace('.npz','_doodles_labelgen.png'), data['doodles'].astype('uint8'))


        ### if only one label
        else:
            print('Only one label')

            if 'orig_image' not in data.keys():
                data['orig_image'] = data['image']

            savez_dict['color_doodles'] = data['color_doodles'].astype('uint8')
            savez_dict['doodles'] = data['doodles'].astype('uint8')
            savez_dict['settings'] = data['settings']
            savez_dict['label'] = data['label'].astype('uint8')
            v = np.unique(data['doodles']).max()
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
            savez_dict['image'] = data['orig_image'].astype('uint8')
            #del data

        np.savez(anno_file.replace('.npz','_labelgen.npz'), **savez_dict )
        del savez_dict
        plt.close('all')


    try:
        lafiles = glob(direc+'/*_labelgen.png')

        for a_file in lafiles:
            shutil.move(a_file,overdir)

    except:
        print('Error when moving labelgen files')


###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:") 
    except getopt.GetoptError:
        print('======================================')
        print('python plot_label_generation.py') 
        print('======================================')
        print('Example usage: python plot_label_generation.py')  
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python plot_label_generation.py ') 
            print('======================================')
            sys.exit()


    #ok, dooo it
    gen_plot_seq()
