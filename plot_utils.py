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


import PIL.Image
import plotly.graph_objects as go
import skimage.util
from plotly.utils import ImageUriValidator

##========================================================
def dummy_fig():
    """ create a dummy figure to be later modified """
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False, scaleanchor="x", showticklabels=False, zeroline=False
    )
    return fig

##========================================================
def pilim(im):
    """ open and return an image file as PIL Image """
    if type(im) == type(str()):
        return PIL.Image.open(im)
    return im

##========================================================
def add_layout_images_to_fig(fig,
    images,
    update_ranges=True):
    """ images is a sequence of PIL Image objects """

    if len(images) <= 0:
        return fig

    for im in images:
        # if image is a path to an image, load the image to get its size
        width, height = pilim(im).size
        # Add images
        fig.add_layout_image(
            dict(
                source=im,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="fill",
                layer="below",
            )
        )
    if update_ranges:
        width, height = [
            max([pilim(im).size[i] for im in images]) for i in range(2)
        ]

        fig.update_xaxes(
            showgrid=False, range=(0, width), showticklabels=False, zeroline=False
        )
        fig.update_yaxes(
            showgrid=False,
            scaleanchor="x",
            range=(height, 0),
            showticklabels=False,
            zeroline=False,
        )

    return fig

##========================================================
def img_array_2_pil(ia):
    """ converst image byte array to PIL Image"""
    ia = skimage.util.img_as_ubyte(ia)
    img = PIL.Image.fromarray(ia)
    return img

##========================================================
def pil2uri(img):
    """ conevrts PIL image to uri"""
    return ImageUriValidator.pil_image_to_uri(img)
