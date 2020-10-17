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

##========================================================
import PIL.Image
import numpy as np
import skimage.util
import skimage.io
import skimage.color
import io, utils
from image_segmentation import segmentation
import plotly.express as px

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
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    colormap=px.colors.qualitative.G10,
    alpha=128,
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

    return np.concatenate(
        (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
    )


##========================================================
def compute_segmentations(
    shapes, median_filter_value,
    img_path="assets/dash-default.jpg",
    segmenter_args={},
    shape_layers=None,
    label_to_colors_args={},
):
    """ segments the image based on the user annotations"""

    # load original image
    img = img_to_ubyte_array(img_path)

    # convert shapes to mask
    shape_args = [
        {"width": img.shape[1], "height": img.shape[0], "shape": shape}
        for shape in shapes
    ]
    if (shape_layers is None) or (len(shape_layers) != len(shapes)):
        shape_layers = [(n + 1) for n, _ in enumerate(shapes)]
    mask = utils.shapes_to_mask(shape_args, shape_layers)

    # do segmentation and return this
    seg, clf = segmentation(img, median_filter_value, mask, **segmenter_args)
    color_seg = label_to_colors(seg, **label_to_colors_args)
    # color_seg is a 3d tensor representing a colored image whereas seg is a
    # matrix whose entries represent the classes
    return (color_seg, seg, clf)


##========================================================
def seg_pil(img, classr):
    """ convert numpy array into PIL Image object """
    classr = np.array(classr)
    classr = skimage.util.img_as_ubyte(classr)
    alpha = (classr[:, :, 3] / 255)[:, :, None]
    classr = classr[:, :, :3]

    return PIL.Image.fromarray(classr)
