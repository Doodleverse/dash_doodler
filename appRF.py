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

import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

from annotations_to_segmentations import *
from plot_utils import *

import io, base64, PIL.Image, json, shutil, os
from glob import glob
from datetime import datetime
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
# from skimage.filters.rank import median
# from skimage.morphology import disk
##========================================================

try:
    os.remove('RandomForestClassifier.pkl.z')
except:
    pass

CURRENT_IMAGE = DEFAULT_IMAGE_PATH = "assets/logos/dash-default.jpg"

DEFAULT_PEN_WIDTH = 2  # gives line width of 2^2 = 4

DEFAULT_DOWNSAMPLE = 10
# DEFAULT_CRF_THETA = 30
# DEFAULT_CRF_MU = 50
DEFAULT_MEDIAN_KERNEL = 5

SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]

# the number of different classes for labels

DEFAULT_LABEL_CLASS = 0

##========================================================

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

class_labels = list(range(NUM_LABEL_CLASSES))


##========================================================
def convert_integer_class_to_color(n):
    return class_label_colormap[n]

def convert_color_class(c):
    return class_label_colormap.index(c)



##========================================================
# files = sorted(glob('assets/*.jpg'))
#
# files = [f for f in files if 'dash' not in f]
#

results_folder = 'results/results'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

try:
    os.mkdir(results_folder)
    print("Results will be written to %s" % (results_folder))
except:
    pass


UPLOAD_DIRECTORY = os.getcwd()+os.sep+"assets"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

files = sorted(glob('assets/*.jpg'))

files = [f for f in files if 'dash' not in f]


##========================================================
def make_and_return_default_figure(
    images=[DEFAULT_IMAGE_PATH],
    stroke_color=convert_integer_class_to_color(DEFAULT_LABEL_CLASS),
    pen_width=DEFAULT_PEN_WIDTH,
    shapes=[],
):

    fig = dummy_fig() #plot_utils.

    add_layout_images_to_fig(fig, images) #plot_utils.

    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": pen_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
            "height": 600
        }
    )

    return fig

##========================================================
def shapes_to_key(shapes):
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


##===============================================================

# UPLOAD_DIRECTORY = os.getcwd()+os.sep+"assets"
#UPLOAD_DIRECTORY = '/media/marda/TWOTB/USGS/SOFTWARE/Projects/dash_doodler/project/to_label'

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id="banner",
            children=[
                html.H1(
            "Doodler: Interactive Segmentation of Imagery",
            id="title",
            className="seven columns",
        ),
        html.Img(id="logo", src=app.get_asset_url("logos/dash-logo-new.png")),
        # html.Div(html.Img(src=app.get_asset_url('logos/dash-logo-new.png'), style={'height':'10%', 'width':'10%'})), #id="logo",

        html.H2(""),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.H2(""),
        html.Ul(id="file-list"),

    ], #children
    ), #div banner id

        html.Div(
            id="main-content",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        dcc.Loading(
                            id="segmentations-loading",
                            type="cube",
                            children=[
                                # Graph
                                dcc.Graph(
                                    id="graph",
                                    figure=make_and_return_default_figure(),
                                    config={
                                        'displayModeBar': 'hover',
                                        "displaylogo": False,
                                        'modeBarOrientation': 'h',
                                        "modeBarButtonsToAdd": [
                                            "drawrect",
                                            "drawopenpath",
                                            "eraseshape",
                                        ]
                                    },
                                ),
                            ],
                        )
                    ],
                    className="ten columns app-background",
                ),

                html.Div(
                       id="right-column",
                       children=[

                        html.H6("Upload images, doodle, download labels. Works best for 3000 x 3000 px images or smaller on most hardware. Info: https://github.com/dbuscombe-usgs/dash_doodler"),
                        html.H6("Select Image"),
                        dcc.Dropdown(
                            id="select-image",
                            optionHeight=15,
                            style={'fontSize': 13},
                            options = [
                                {'label': image.split('assets/')[-1], 'value': image } \
                                for image in files
                            ],

                            value='assets/logos/dash-default.jpg', #
                            multi=False,
                        ),


                        html.H6("Label class"),
                        # Label class chosen with buttons
                        html.Div(
                            id="label-class-buttons",
                            children=[
                                html.Button(
                                    #"%2d" % (n,),
                                    "%s" % (class_label_names[n],),
                                    id={"type": "label-class-button", "index": n},
                                    style={"background-color": convert_integer_class_to_color(c)},
                                )
                                for n, c in enumerate(class_labels)
                            ],
                        ),

                        html.H6(id="pen-width-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="pen-width",
                            min=1,
                            max=6,
                            step=0.5,
                            value=DEFAULT_PEN_WIDTH,
                        ),


                        # Indicate showing most recently computed segmentation
                        dcc.Checklist(
                            id="rf-show-segmentation",
                            options=[
                                {
                                    "label": "Compute/Show segmentation",
                                    "value": "Show segmentation",
                                }
                            ],
                            value=[],
                        ),


                        html.H6(id="median-filter-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="median-filter",
                            min=0.1,
                            max=100,
                            step=0.1,
                            value=DEFAULT_MEDIAN_KERNEL,
                        ),

                        html.H6("Image Feature Extraction:"),
                        dcc.Checklist(
                            id="segmentation-features",
                            options=[
                                {"label": l.capitalize(), "value": l}
                                for l in SEG_FEATURE_TYPES
                            ],
                            value=["intensity", "texture"],
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.H6(id="sigma-display"),
                        dcc.RangeSlider(
                            id="sigma-range-slider",
                            min=1,
                            max=30,
                            step=1,
                            value=[1, 16],
                        ),

                        html.H6(id="downsample-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="downsample-slider",
                            min=2,
                            max=30,
                            step=1,
                            value=DEFAULT_DOWNSAMPLE,
                        ),

                        html.A(
                            id="download-image",
                            download="classified-image-"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".png",
                            children=[
                                html.Button(
                                    "Download Label Image",
                                    id="download-image-button",
                                )
                            ],
                        ),
                    ],
                    className="three columns app-background",
                ),
            ],
            className="ten columns",
        ), #main content Div

        html.Div(
            id="no-display",
            children=[
                dcc.Store(id="image-list-store", data=[]),
                # Store for user created masks
                # data is a list of dicts describing shapes
                dcc.Store(id="masks", data={"shapes": []}),
                # Store for storing segmentations from shapes
                # the keys are hashes of shape lists and the data are pngdata
                # representing the corresponding segmentation
                # this is so we can download annotations and also not recompute
                # needlessly old segmentations
                dcc.Store(id="segmentation", data={}),
                dcc.Store(id="classified-image-store", data=""),
            ],
        ), #nos-display div

    ], #children
) #app layout

##============================================================
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            if 'jpg' in filename:
                files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

    #
# median_filter_value
##========================================================
def show_segmentation(image_path,
    mask_shapes,
    segmenter_args,
    median_filter_value,
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
    results_folder,
    downsample_value,
    crf_downsample_factor):
    """ adds an image showing segmentations to a figure's layout """

    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [convert_color_class(shape["line"]["color"]) + 1 for shape in mask_shapes]

    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }

    segimg, seg, img = compute_segmentations(
        mask_shapes, None, None, #crf_theta_slider_value,crf_mu_slider_value,
        results_folder,  median_filter_value, downsample_value,
        crf_downsample_factor, callback_context,
        img_path=image_path,
        segmenter_args=segmenter_args,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
    )

    # get the classifier that we can later store in the Store
    segimgpng = img_array_2_pil(segimg) #plot_utils.

    return (segimgpng, seg, img)

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

def look_up_seg(d, key):
    """ Returns a PIL.Image object """
    data = d[key]
    img_bytes = base64.b64decode(data)
    img = PIL.Image.open(io.BytesIO(img_bytes))
    return img

# ##========================================================
# Input('crf-mu-slider', "value"),
    # Output("theta-display", "children"),
    # Output("mu-display", "children"),

@app.callback(
    [
    Output("select-image","options"),
    Output("graph", "figure"),
    Output("image-list-store", "data"),
    Output("masks", "data"),
    Output("segmentation", "data"),
    Output("pen-width-display", "children"),
    Output("sigma-display", "children"),
    Output("median-filter-display", "children"),
    Output("downsample-display", "children"),
    Output("classified-image-store", "data"),
    ],
    [
    Input("upload-data", "filename"),
    Input("upload-data", "contents"),
    Input("graph", "relayoutData"),
    Input(
        {"type": "label-class-button", "index": dash.dependencies.ALL},
        "n_clicks_timestamp",
    ),
    Input("pen-width", "value"),
    Input("rf-show-segmentation", "value"),
    Input("median-filter", "value"),
    Input("segmentation-features", "value"),
    Input("sigma-range-slider", "value"),
    Input("downsample-slider", "value"),
    Input("select-image", "value"),
    ],
    [
    State("image-list-store", "data"),
    State("masks", "data"),
    State("segmentation", "data"),
    State("classified-image-store", "data"),
    ],
)

# ##========================================================
    # crf_theta_slider_value,
    # crf_mu_slider_value,

def update_output(
    uploaded_filenames,
    uploaded_file_contents,
    graph_relayoutData,
    any_label_class_button_value,
    pen_width_value,
    show_segmentation_value,
    median_filter_value,
    segmentation_features_value,
    sigma_range_slider_value,
    downsample_value,
    select_image_value,
    image_list_data,
    masks_data,
    segmentation_data,
    segmentation_store_data,
    ):
    """Save uploaded files and regenerate the file list."""

    callback_context = [p["prop_id"] for p in dash.callback_context.triggered][0]
    print(callback_context)

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)
    else:
        image_list_data = []

    # if new image selected, compare lists of files versus done, and populates menu only with to-do
    if callback_context == "select-image.value":
        files = uploaded_files()
        # get list of files that are different between options (all files) and image_list_data (already labeled)
        image_list_data_nofolder = [f.split('assets/')[-1] for f in image_list_data]
        files = list(set(files) - set(image_list_data_nofolder))

        options = [{'label': image.split('assets/')[-1], 'value': image } for image in files]
        # print(options)
    else:
        files = uploaded_files()
        options = [{'label': image.split('assets/')[-1], 'value': image } for image in files]

    if 'assets' not in select_image_value:
        select_image_value = 'assets'+os.sep+select_image_value
        # print(select_image_value)

    if callback_context == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"] = graph_relayoutData["shapes"]
        else:
            return dash.no_update

    elif callback_context == "select-image.value":
       masks_data={"shapes": []}
       segmentation_data={}

    pen_width = int(round(2 ** (pen_width_value)))

    # find label class value by finding button with the greatest n_clicks
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    fig = make_and_return_default_figure(
        images = [select_image_value],
        stroke_color=convert_integer_class_to_color(label_class_value),
        pen_width=pen_width,
        shapes=masks_data["shapes"],
    )

    if ("Show segmentation" in show_segmentation_value) and (
        len(masks_data["shapes"]) > 0):
        # to store segmentation data in the store, we need to base64 encode the
        # PIL.Image and hash the set of shapes to use this as the key
        # to retrieve the segmentation data, we need to base64 decode to a PIL.Image
        # because this will give the dimensions of the image
        sh = shapes_to_key(
            [
                masks_data["shapes"],
                segmentation_features_value,
                sigma_range_slider_value,
            ]
        )

        segimgpng = None
        if 'median'not  in callback_context:
            dict_feature_opts = {
                key: (key in segmentation_features_value)
                for key in SEG_FEATURE_TYPES
            }

            dict_feature_opts["sigma_min"] = sigma_range_slider_value[0]
            dict_feature_opts["sigma_max"] = sigma_range_slider_value[1]

            if len(segmentation_features_value) > 0:
                segimgpng, seg, img = show_segmentation(
                    [select_image_value], masks_data["shapes"], dict_feature_opts, median_filter_value, callback_context,
                    None, None, results_folder, downsample_value, None,
                )

                if type(select_image_value) is list:
                    imsave(select_image_value[0].replace('assets',results_folder).replace('.jpg','_label.png'),
                            label_to_colors(seg-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
                else:
                    imsave(select_image_value.replace('assets',results_folder).replace('.jpg','_label.png'),
                            label_to_colors(seg-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

                if type(select_image_value) is list:
                    imsave(select_image_value[0].replace('assets',results_folder).replace('.jpg','_label_greyscale.png'), seg)
                else:
                    imsave(select_image_value.replace('assets',results_folder).replace('.jpg','_label_greyscale.png'), seg)
                del img, seg

                segmentation_data = shapes_seg_pair_as_dict(
                    segmentation_data, sh, segimgpng
                )
                try:
                    segmentation_store_data = pil2uri(
                        seg_pil(
                            select_image_value, segimgpng, do_alpha=True
                        ) #plot_utils.
                    )
                except:
                    segmentation_store_data = pil2uri(
                        seg_pil(
                            PIL.Image.open(select_image_value), segimgpng, do_alpha=True
                        ) #plot_utils.
                    )
        # except ValueError:
        #     # if segmentation fails, draw nothing
        #     pass


        images_to_draw = []

        if segimgpng is not None:
            images_to_draw = [segimgpng]

        fig = add_layout_images_to_fig(fig, images_to_draw) #plot_utils.

        show_segmentation_value = []

        image_list_data.append(select_image_value)
        # print(image_list_data)


    if len(files) == 0:
        return [
        options,
        fig,
        image_list_data,
        masks_data,
        segmentation_data,
        "Pen width: %d" % (pen_width,),
        "Blurring parameter for RF feature extraction: %d, %d" % (sigma_range_slider_value[0], sigma_range_slider_value[1]),
        "Median filter kernel radius: %d" % (median_filter_value,),
        "RF downsample factor: %d" % (downsample_value,),
        segmentation_store_data,
        ]
    else:
        return [
        options,
        fig,
        image_list_data,
        masks_data,
        segmentation_data,
        "Pen width: %d" % (pen_width,),
        "Blurring parameter for RF feature extraction: %d, %d" % (sigma_range_slider_value[0], sigma_range_slider_value[1]),
        "Median filter kernel radius: %d" % (median_filter_value,),
        "RF downsample factor: %d" % (downsample_value,),
        segmentation_store_data,
        ]


# "Blurring parameter for CRF image feature extraction: %d" % (crf_theta_slider_value,),
# "CRF color class difference tolerance parameter: %d" % (crf_mu_slider_value,),
#
# "Blurring parameter for CRF image feature extraction: %d" % (crf_theta_slider_value,),
# "CRF color class difference tolerance parameter: %d" % (crf_mu_slider_value,),

##========================================================

# set the download url to the contents of the classified-image-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_image_store_data) {
    return the_image_store_data;
}
""",
    Output("download-image", "href"),
    [Input("classified-image-store", "data")],
)


if __name__ == "__main__":
    app.run_server() #debug=True, port=8888)

        # if callback_context == "median-filter.value":
        #     if sh in segmentation_data.keys():
        #         if median_filter_value>1: #"Apply Median Filter" in median_filter_value:
        #             print("applying median filter:")
        #             mask = imread(select_image_value.replace('assets',results_folder).replace('.jpg','_label_greyscale.png'))
        #             mask = median(mask, disk(median_filter_value)).astype(np.uint8)
        #             # print(segimgpng.shape)
        #
        #             label_to_colors_args = {
        #                 "colormap": class_label_colormap,
        #                 "color_class_offset": -1,
        #             }
        #
        #             try:
        #                 segimgpng = label_to_colors(mask, img[:,:,0]==0, alpha=0, **label_to_colors_args, do_alpha=True)
        #             except:
        #                 img = img_to_ubyte_array(select_image_value)
        #                 segimgpng = label_to_colors(mask, img[:,:,0]==0, alpha=0, **label_to_colors_args, do_alpha=True)
        #
        #             # segimgpng = np.dstack((segimgpng, np.zeros(segimgpng.shape[:1])))
        #             print(segimgpng.shape)
        #             #segimgpng = PIL.Image.fromarray(np.uint8(segimgpng))
        #
        #             segimgpng = img_array_2_pil(segimgpng) #plot_utils.
        #             # print(segimgpng.shape)
        #
        #             if type(select_image_value) is list:
        #                 imsave(select_image_value[0].replace('assets',results_folder).replace('.jpg','_label.png'), label_to_colors(mask-1, img[:,:,0]==0, class_label_colormap, do_alpha=False))
        #             else:
        #                 imsave(select_image_value.replace('assets',results_folder).replace('.jpg','_label.png'), label_to_colors(mask-1, img[:,:,0]==0, class_label_colormap, do_alpha=False))
        #
        #             if type(select_image_value) is list:
        #                 imsave(select_image_value[0].replace('assets',results_folder).replace('.jpg','_label_greyscale.png'), mask)
        #             else:
        #                 imsave(select_image_value.replace('assets',results_folder).replace('.jpg','_label_greyscale.png'), mask)
        #             del mask
        #
        #             segmentation_data = shapes_seg_pair_as_dict(
        #                 segmentation_data, sh, segimgpng
        #             )
        #
        #             segmentation_store_data = pil2uri(
        #                 seg_pil(
        #                     select_image_value, segimgpng, do_alpha=True
        #                 ) #plot_utils.
        #             )
        #             images_to_draw = []
        #
        #             if segimgpng is not None:
        #                 images_to_draw = [segimgpng]
        #
        #             fig = add_layout_images_to_fig(fig, images_to_draw) #plot_utils.
        #
        #             show_segmentation_value = []


        # if sh in segmentation_data.keys():
        #     segimgpng = look_up_seg(segmentation_data, sh)
        # else:
