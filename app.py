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
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plot_utils
from annotations_to_segmentations import (
    compute_segmentations,
    seg_pil,
)
import io, base64, PIL.Image, json, shutil, os
from glob import glob
from datetime import datetime

##========================================================
CURRENT_IMAGE = DEFAULT_IMAGE_PATH = "assets/dash-default.jpg"

DEFAULT_PEN_WIDTH = 2  # gives line width of 2^2 = 4

SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]

# the number of different classes for labels

DEFAULT_LABEL_CLASS = 0
class_label_colormap = px.colors.qualitative.G10


with open('classes.txt') as f:
    classes = f.readlines()

class_label_names = [c.strip() for c in classes]

# class_label_names = ['deep', 'white', 'shallow', 'dry']

NUM_LABEL_CLASSES = len(class_label_names)

# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

class_labels = list(range(NUM_LABEL_CLASSES))


##========================================================
def convert_integer_class_to_color(n):
    return class_label_colormap[n]

def convert_color_class(c):
    return class_label_colormap.index(c)


files = sorted(glob('assets/*.jpg'))

files = [f for f in files if 'dash' not in f]

app = dash.Dash(__name__)
server = app.server

##========================================================
def make_and_return_default_figure(
    images=[DEFAULT_IMAGE_PATH],
    stroke_color=convert_integer_class_to_color(DEFAULT_LABEL_CLASS),
    pen_width=DEFAULT_PEN_WIDTH,
    shapes=[],
):

    fig = plot_utils.dummy_fig()

    plot_utils.add_layout_images_to_fig(fig, images)

    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": pen_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
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

##========================================================

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
                html.Img(id="logo", src=app.get_asset_url("dash-logo-new.png"),),
            ],
            className="ten columns app-background",
        ),

        html.Div(
            id="description",
            children=[
                html.P(
                    'Make some annotations on the picture using different colors each of the label classes present. '+
                    'Then select "Show segmentation" to see the segmentation. Play with the features, median filter, and blurring parameter (if you like). '+
                    'You may add more annotations to clarify where the classifier was in error. Each time you make a change, wait for the classification to update.',
                    className="ten columns",
                ),
                html.Img(
                    id="example-image",
                    src="assets/dash-segmentation_img_example_marks.jpg",
                    className="two columns",
                ),
            ],
            className="ten columns app-background",
        ),
        html.Div(
            id="main-content",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        dcc.Loading(
                            id="segmentations-loading",
                            type="circle",
                            children=[
                                # Graph
                                dcc.Graph(
                                    id="graph",
                                    figure=make_and_return_default_figure(),
                                    config={
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
                    className="eight columns app-background",
                ),

                html.Div(
                    id="right-column",
                    children=[

                       html.H6("Select Image"),
                       dcc.Dropdown(
                            id="select-image",

                            options = [
                                {'label': image.split('assets/')[-1], 'value': image } \
                                for image in files
                            ],

                            value='assets/dash-default.jpg', #
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
                            min=0,
                            max=6,
                            step=0.1,
                            value=DEFAULT_PEN_WIDTH,
                        ),


                        html.H6("Image segmentation:"),

                        # Indicate showing most recently computed segmentation
                        dcc.Checklist(
                            id="show-segmentation",
                            options=[
                                {
                                    "label": "Compute and show segmentation",
                                    "value": "Show segmentation",
                                }
                            ],
                            value=[],
                        ),


                        dcc.Checklist(
                            id="median-filter",
                            options=[
                                {
                                    "label": "Apply median filter",
                                    "value": "Apply Median Filter",
                                }
                            ],
                            value=["Apply Median Filter"],
                        ),


                        html.H6("Image Feature Extraction:"),
                        dcc.Checklist(
                            id="segmentation-features",
                            options=[
                                {"label": l.capitalize(), "value": l}
                                for l in SEG_FEATURE_TYPES
                            ],
                            value=["intensity", "texture"],
                        ),
                        html.H6("Blurring parameter for image feature extraction:"),
                        dcc.RangeSlider(
                            id="sigma-range-slider",
                            min=0.01,
                            max=20,
                            step=0.01,
                            value=[2, 16],
                        ),

                        # We use this pattern because we want to be able to download the
                        # annotations by clicking on a button
                        # html.A(
                        #     id="download",
                        #     download="annotations-"+datetime.now().strftime("%d-%m-%Y-%H-%M")+".json",
                        #     children=[
                        #         html.Button(
                        #             "Download annotations", id="download-button"
                        #         ),
                        #         html.Span(
                        #             " ",
                        #             className="tooltiptext",
                        #         ),
                        #     ],
                        #     className="tooltip",
                        # ),
                        # html.A(
                        #     id="download-image",
                        #     download="classified-image-"+datetime.now().strftime("%d-%m-%Y-%H-%M")+".png",
                        #     children=[
                        #         html.Button(
                        #             "Download classified image",
                        #             id="download-image-button",
                        #         )
                        #     ],
                        # ),
                    ],
                    className="four columns app-background",
                ),
            ],
            className="ten columns",
        ),
        html.Div(
            id="no-display",
            children=[
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
        ),
    ],
)

##========================================================
def show_segmentation(image_path, mask_shapes, segmenter_args, median_filter_value):
    """ adds an image showing segmentations to a figure's layout """

    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [convert_color_class(shape["line"]["color"]) + 1 for shape in mask_shapes]

    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }

    segimg, _, clf = compute_segmentations(
        mask_shapes, median_filter_value,
        img_path=image_path,
        segmenter_args=segmenter_args,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
    )

    # get the classifier that we can later store in the Store
    segimgpng = plot_utils.img_array_2_pil(segimg)

    return (segimgpng)


##========================================================
@app.callback(
    [
        Output("graph", "figure"),
        Output("masks", "data"),
        Output("segmentation", "data"),
        Output("pen-width-display", "children"),
        Output("classified-image-store", "data"),
    ],
    [
        Input("graph", "relayoutData"),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("pen-width", "value"),
        Input("show-segmentation", "value"),
        Input("median-filter", "value"),
        Input("segmentation-features", "value"),
        Input("sigma-range-slider", "value"),
        Input("select-image", "value"),
    ],
    [
        State("masks", "data"),
        State("segmentation", "data"),
        State("classified-image-store", "data"),
    ],
)

##========================================================
def annotation_react_enact(
    graph_relayoutData,
    any_label_class_button_value,
    pen_width_value,
    show_segmentation_value,
    median_filter_value,
    segmentation_features_value,
    sigma_range_slider_value,
    select_image_value,
    masks_data,
    segmentation_data,
    segmentation_store_data,
):

    callback_context = [p["prop_id"] for p in dash.callback_context.triggered][0]
    print(callback_context)

    if callback_context == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"] = graph_relayoutData["shapes"]
        else:
            return dash.no_update

    elif callback_context == "select-image.value":
       masks_data={"shapes": []}


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
        try:
            dict_feature_opts = {
                key: (key in segmentation_features_value)
                for key in SEG_FEATURE_TYPES
            }

            dict_feature_opts["sigma_min"] = sigma_range_slider_value[0]
            dict_feature_opts["sigma_max"] = sigma_range_slider_value[1]

            if len(segmentation_features_value) > 0:
                segimgpng = show_segmentation(
                    [select_image_value], masks_data["shapes"], dict_feature_opts, median_filter_value
                )

                segmentation_data = shapes_seg_pair_as_dict(
                    segmentation_data, sh, segimgpng
                )
                try:
                    segmentation_store_data = plot_utils.pil2uri(
                        seg_pil(
                            select_image_value, segimgpng
                        )
                    )
                except:
                    segmentation_store_data = plot_utils.pil2uri(
                        seg_pil(
                            PIL.Image.open(select_image_value), segimgpng
                        )
                    )
        except ValueError:
            # if segmentation fails, draw nothing
            pass


        images_to_draw = []
        if segimgpng is not None:
            images_to_draw = [segimgpng]

        fig = plot_utils.add_layout_images_to_fig(fig, images_to_draw)

    return (
        fig,
        masks_data,
        segmentation_data,
        "Pen width: %d" % (pen_width,),
        segmentation_store_data,
    )

##========================================================

# set the download url to the contents of the classifier-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("segmentation", "data")],
)

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

##========================================================
if __name__ == "__main__":
    app.run_server()
