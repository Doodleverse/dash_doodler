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

##========================================================
#========================================================
## ``````````````````````````` imports
##========================================================

import numpy as np
import PIL.Image, skimage.util, skimage.io, skimage.color
from PIL import ExifTags

import io, os, psutil, logging, base64, json
from datetime import datetime
from image_segmentation import segmentation
import plotly.express as px
from skimage.io import imsave, imread
from cairosvg import svg2png
from datetime import datetime
from app_funcs import *
from plot_utils import *

    # gt_prob,

##========================================================
def show_segmentation(image_path,
    mask_shapes,
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
    results_folder,
    rf_downsample_value,
    crf_downsample_factor,
    null,
    my_id_value,
    n_sigmas,
    multichannel,
    intensity,
    edges,
    texture,
    class_label_colormap
    ):

    """ adds an image showing segmentations to a figure's layout """

    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [convert_color_class(class_label_colormap,shape["line"]["color"]) + 1 for shape in mask_shapes]

    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }

    sigma_min=1; sigma_max=16

    segimg, seg, img, color_doodles, doodles = compute_segmentations(
        mask_shapes, crf_theta_slider_value,crf_mu_slider_value,
        rf_downsample_value, #results_folder, 
        crf_downsample_factor,  n_sigmas, #my_id_value, callback_context, gt_prob,
        multichannel, intensity, edges, texture, sigma_min, sigma_max,
        img_path=image_path,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
    )

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Showing segmentation on screen ...')
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    logging.info('... converting to PIL array ...')
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))
    # get the classifier that we can later store in the Store
    segimgpng = img_array_2_pil(segimg) #plot_utils.

    return (segimgpng, seg, np.squeeze(img), color_doodles, doodles )

##========================================================
def img_array_2_pil(ia):
    """ converst image byte array to PIL Image"""
    ia = skimage.util.img_as_ubyte(ia)
    img = PIL.Image.fromarray(ia)
    return img

##========================================================
def convert_integer_class_to_color(class_label_colormap,n):
    """
    class to color
    """
    return class_label_colormap[n]

##========================================================
def convert_color_class(class_label_colormap,c):
    """
    color to class
    """
    return class_label_colormap.index(c)

##========================================================
def shapes_to_key(shapes):
    """
    convert shapes to json
    """
    return json.dumps(shapes)

##========================================================
def shapes_seg_pair_as_dict(d, key, seg, remove_old=True):
    """
    Stores shapes and segmentation pair in dict d
    seg is a PIL.Image object
    if remove_old True, deletes all the old keys and values.
    """
    bytes_to_encode = io.BytesIO()
    seg.save(bytes_to_encode, format="png")
    bytes_to_encode.seek(0)

    data = base64.b64encode(bytes_to_encode.read()).decode()

    if remove_old:
        return {key: data}
    d[key] = data

    return d

##========================================================
def shape_to_svg_code(shape, fig=None, width=None, height=None):
    """
    fig is the plotly.py figure which shape resides in (to get width and height)
    and shape is one of the shapes the figure contains.
    """
    if fig is not None:
        # get width and height
        wrange = next(fig.select_xaxes())["range"]
        hrange = next(fig.select_yaxes())["range"]
        width, height = [max(r) - min(r) for r in [wrange, hrange]]
    else:
        if width is None or height is None:
            raise ValueError("If fig is None, you must specify width and height")
    fmt_dict = dict(
        width=width,
        height=height,
        stroke_color=shape["line"]["color"],
        stroke_width=shape["line"]["width"],
        path=shape["path"],
    )
    return """
<svg
    width="{width}"
    height="{height}"
    viewBox="0 0 {width} {height}"
>
<path
    stroke="{stroke_color}"
    stroke-width="{stroke_width}"
    d="{path}"
    fill-opacity="0"
/>
</svg>
""".format(
        **fmt_dict
    )

##========================================================
def shape_to_png(fig=None, shape=None, width=None, height=None, write_to=None):
    """
    Like svg2png, if write_to is None, returns a bytestring. If it is a path
    to a file it writes to this file and returns None.
    """
    svg_code = shape_to_svg_code(fig=fig, shape=shape, width=width, height=height)
    r = svg2png(bytestring=svg_code, write_to=write_to)
    return r

##========================================================
def shapes_to_mask(shape_args, shape_layers):
    """
    Returns numpy array (type uint8) with number of rows equal to maximum height
    of all shapes's bounding boxes and number of columns equal to their number
    of rows.
    shape_args is a list of dictionaries whose keys are the parameters to the
    shape_to_png function.
    The mask is taken to be all the pixels that are non-zero in the resulting
    image from rendering the shape.
    shape_layers is either a number or an array
    if a number, all the layers have the same number in the mask
    if an array, must be the same length as shape_args and each entry is an
    integer in [0...255] specifying the layer number. Note that the convention
    is that 0 means no mask, so generally the layer numbers will be non-zero.
    """

    images = []
    for sa in shape_args:
        pngbytes = shape_to_png(**sa)
        images.append(PIL.Image.open(io.BytesIO(pngbytes)))

    mwidth, mheight = [max([im.size[i] for im in images]) for i in range(2)]
    mask = np.zeros((mheight, mwidth), dtype=np.uint8)

    if type(shape_layers) != type(list()):
        layer_numbers = [shape_layers for _ in shape_args]
    else:
        layer_numbers = shape_layers

    imarys = []
    for layer_num, im in zip(layer_numbers, images):
        # layer 0 is reserved for no mask
        imary = skimage.util.img_as_ubyte(np.array(im))
        imary = np.sum(imary, axis=2)
        imary.resize((mheight, mwidth))
        imarys.append(imary)
        mask[imary != 0] = layer_num

    return mask

##========================================================
def img_to_ubyte_array(img):
    """
    PIL.Image.open is used so that a io.BytesIO object containing the image data
    can be passed as img and parsed into an image. Passing a path to an image
    for img will also work.
    """
    # try:
    #    img = np.array(PIL.Image.open(img))
    #    if np.ndim(img)>3:
    #        img = img[:,:,:3]
    #    ret = skimage.util.img_as_ubyte(img)
    # except:
    #    img = np.array(PIL.Image.open(img[0]))
    #    if np.ndim(img)>3:
    #        img = img[:,:,:3]
    #    ret = skimage.util.img_as_ubyte(img)

    # if np.ndim(ret)>3:
    #     ret = ret[:,:,:3]

    # return ret

    try:
        image = PIL.Image.open(img)		 	
    except:
        image = PIL.Image.open(img[0])		 	

    for orientation in ExifTags.TAGS.keys():
        # if ExifTags.TAGS[orientation]=='Orientation':
        #    break
        try:
            exif=dict(image._getexif().items())
            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
                # print("Image rotated 180 deg")
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
                # print("Image rotated 270 deg")
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)
                # print("Image rotated 90 deg")
        except:
            # print('no exif')
            pass

    img = np.array(image)
    if np.ndim(img)>3:
        img = img[:,:,:3]
    ret = skimage.util.img_as_ubyte(img)

    if np.ndim(ret)>3:
        ret = ret[:,:,:3]

    return ret

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    mask,
    alpha,#=128,
    colormap,#=class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,#=0,
    do_alpha,#=True
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

    cimg[mask==1] = (0,0,0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg

    gt_prob,
    # results_folder,
    # my_id_value,
    # callback_context,
##========================================================
def compute_segmentations(
    shapes,
    crf_theta_slider_value,
    crf_mu_slider_value,
    rf_downsample_value,
    crf_downsample_factor,
    n_sigmas,
    multichannel,
    intensity,
    edges,
    texture,
    sigma_min,
    sigma_max,
    img_path="assets/logos/dash-default.jpg",
    shape_layers=None,
    label_to_colors_args={},
    ):
    """
    segments the image based on the user annotations
    calls segmentation() from image_segmentation
    """

    # load original image and convert to unit8
    img = img_to_ubyte_array(img_path)

    # convert shapes to mask
    shape_args = [
        {"width": img.shape[1], "height": img.shape[0], "shape": shape}
        for shape in shapes
    ]
    if (shape_layers is None) or (len(shape_layers) != len(shapes)):
        shape_layers = [(n + 1) for n, _ in enumerate(shapes)]
    mask = shapes_to_mask(shape_args, shape_layers) #utils.

    if np.ndim(img)==3:
        color_annos = label_to_colors(mask, img[:,:,0]==0, alpha=128, do_alpha=True, **label_to_colors_args)
    else:
        color_annos = label_to_colors(mask, img==0, alpha=128, do_alpha=True, **label_to_colors_args)

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))
    logging.info('Calling segmentation function')

    seg = segmentation(img, #img_path, results_folder, callback_context, 
                       crf_theta_slider_value, crf_mu_slider_value,  rf_downsample_value, #median_filter_value,
                       crf_downsample_factor, mask, n_sigmas, multichannel, intensity, edges, texture, #gt_prob, 
                       sigma_min, sigma_max)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Segmentation computed')

    #print(np.unique(seg))
    if np.ndim(img)==3:
        color_seg = label_to_colors(seg, img[:,:,0]==0, alpha=128, do_alpha=True, **label_to_colors_args)
    else:
        color_seg = label_to_colors(seg, img==0, alpha=128, do_alpha=True, **label_to_colors_args)

    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Color segmentation computed')
    logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    # color_seg is a 3d tensor representing a colored image whereas seg is a
    # matrix whose entries represent the classes
    return (color_seg, seg, img, color_annos[:,:,:3], mask ) #colored image, label image, input image, color annotations, greyscale annotations


##========================================================
def seg_pil(img, classr, do_alpha=False):
    """ convert numpy array into PIL Image object """
    classr = np.array(classr)
    classr = skimage.util.img_as_ubyte(classr)
    if do_alpha is True:
        alpha = (classr[:, :, 3] / 255)[:, :, None]
        classr = classr[:, :, :3]

    return PIL.Image.fromarray(classr)
