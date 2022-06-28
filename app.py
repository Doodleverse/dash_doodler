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
## ``````````````````````````` local imports
# allows loading of functions from the src directory
import sys,os
# sys.path.insert(1, 'app_files'+os.sep+'src')
# from annotations_to_segmentations import *
from doodler_engine.annotations_to_segmentations import *

#========================================================
## ``````````````````````````` imports
##========================================================

## dash/plotly/flask
import plotly.express as px
import plotly.graph_objects as go
# import skimage.util
from plotly.utils import ImageUriValidator


import dash
from dash.dependencies import Input, Output, State
try:
    from dash import html
except:
    import dash_html_components as html

try:
    from dash import dcc
except:
    import dash_core_components as dcc

from flask import Flask
from flask_caching import Cache

#others
import base64, PIL.Image, json, shutil, time, logging, psutil
from datetime import datetime


##========================================================
def make_and_return_default_figure(
    images,#=[DEFAULT_IMAGE_PATH],
    stroke_color,#=convert_integer_class_to_color(class_label_colormap,DEFAULT_LABEL_CLASS),
    pen_width,#=DEFAULT_PEN_WIDTH,
    shapes#=[],
):
    """
    create and return the default Dash/plotly figure object
    """
    fig = dummy_fig() #plot_utils.

    add_layout_images_to_fig(fig, images) #plot_utils.

    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": pen_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
            "height": 650
        }
    )

    return fig


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
def pil2uri(img):
    """ conevrts PIL image to uri"""
    return ImageUriValidator.pil_image_to_uri(img)


##========================================================
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

#========================================================
## defaults
#========================================================

DEFAULT_IMAGE_PATH = "assets"+os.sep+"logos"+os.sep+"dash-default.jpg"

try:
    from my_defaults import *
    print('Hyperparameters imported from my_defaults.py')
except:
    from doodler_engine.defaults import *
    print('Default hyperparameters imported from src/my_defaults.py')

#========================================================
## logs
#========================================================

logging.basicConfig(filename=os.getcwd()+os.sep+'app_files'+os.sep+'logs'+
                    os.sep+datetime.now().strftime("%Y-%m-%d-%H-%M")+'.log',
                    level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

#========================================================
## folders
#========================================================

UPLOAD_DIRECTORY = os.getcwd()+os.sep+"assets"
LABELED_DIRECTORY = os.getcwd()+os.sep+"labeled"
results_folder = 'results'+os.sep+'results'+datetime.now().strftime("%Y-%m-%d-%H-%M")

try:
    os.mkdir(results_folder)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info("Folder created: %s" % (results_folder))
except:
    pass

logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
logging.info("Results will be written to %s" % (results_folder))

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logging.info('Made the directory '+UPLOAD_DIRECTORY)


##========================================================
## classes
#========================================================

# the number of different classes for labels
DEFAULT_LABEL_CLASS = 0

try:
    with open('classes.txt') as f:
        classes = f.readlines()
except: #in case classes.txt does not exist
    print("classes.txt not found or badly formatted. \
            Exit the program and fix the classes.txt file ... \ otherwise, will continue using default classes. ")
    classes = ['water', 'land']

class_label_names = [c.strip() for c in classes]

NUM_LABEL_CLASSES = len(class_label_names)

#========================================================
## colormap
#========================================================

if NUM_LABEL_CLASSES<=10:
    class_label_colormap = px.colors.qualitative.G10
else:
    class_label_colormap = px.colors.qualitative.Light24

# we can't have fewer colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

class_labels = list(range(NUM_LABEL_CLASSES))

logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
logging.info('loaded class labels:')
for f in class_label_names:
    logging.info(f)

#========================================================
## image asset files
#========================================================
files = get_asset_files()

logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
logging.info('loaded files:')
for f in files:
    logging.info(f)

##========================================================
# app, server, and cache
#========================================================

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)

#app = dash.Dash(__name__)
#server = app.server
app.config.suppress_callback_exceptions=True
# app = dash.Dash(__name__)
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server
# app.config.suppress_callback_exceptions = True

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'app_files'+os.sep+'cache-directory'
})

##========================================================
## app layout
##========================================================
app.layout = html.Div(
    id="app-container",
    children=[
        #========================================================
        ## tab 1
        #========================================================

        html.Div(
            id="banner",
            children=[
                html.H1(
            "Doodler: Interactive Image Segmentation",
            id="title",
            className="seven columns",
        ),

        html.Img(id="logo", src=app.get_asset_url("logos"+os.sep+"dash-logo-new.png")),
        # html.Div(html.Img(src=app.get_asset_url('logos/dash-logo-new.png'), style={'height':'10%', 'width':'10%'})), #id="logo",

        html.H2(""),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["    "] #(Label all classes that are present, in all regions of the image those classes occur)
            ),
            style={
                "width": "100%",
                "height": "30px",
                "lineHeight": "70px",
                "borderWidth": "1px",
                "borderStyle": "none",
                "borderRadius": "1px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.H2(""),
        html.Ul(id="file-list"),

    ], #children
    ), #div banner id

    dcc.Tabs([
        dcc.Tab(label='Imagery and Controls', children=[

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
                                    figure=make_and_return_default_figure(
                                        images=[DEFAULT_IMAGE_PATH],
                                        stroke_color=convert_integer_class_to_color(class_label_colormap,DEFAULT_LABEL_CLASS),
                                        pen_width=DEFAULT_PEN_WIDTH,
                                        shapes=[],
                                    ),
                                    config={
                                        'displayModeBar': 'hover',
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": [
                                            "toImage",
                                            "hoverClosestCartesian",
                                            "hoverCompareCartesian",
                                            "toggleSpikelines",
                                        ],
                                        "modeBarButtonsToAdd": [
                                            "drawopenpath",
                                            "eraseshape",
                                        ]
                                    },
                                ),
                            ],
                        ),

                    ],
                    className="ten columns app-background",
                ),

                html.Div(
                       id="right-column",
                       children=[


                        html.H6("Label class"),
                        # Label class chosen with buttons
                        html.Div(
                            id="label-class-buttons",
                            children=[
                                html.Button(
                                    #"%2d" % (n,),
                                    "%s" % (class_label_names[n],),
                                    id={"type": "label-class-button", "index": n},
                                    style={"background-color": convert_integer_class_to_color(class_label_colormap,c)},
                                )
                                for n, c in enumerate(class_labels)
                            ],
                        ),

                        html.H6(id="pen-width-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="pen-width",
                            min=0,
                            max=5,
                            step=1,
                            value=DEFAULT_PEN_WIDTH,
                        ),


                        # Indicate showing most recently computed segmentation
                        dcc.Checklist(
                            id="crf-show-segmentation",
                            options=[
                                {
                                    "label": "Compute/Show segmentation",
                                    "value": "Show segmentation",
                                }
                            ],
                            value=[],
                        ),

                        dcc.Markdown(
                            ">Post-processing settings"
                        ),

                        html.H6(id="theta-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="crf-theta-slider",
                            min=1,
                            max=100,
                            step=1,
                            value=DEFAULT_CRF_THETA,
                        ),

                        html.H6(id="mu-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="crf-mu-slider",
                            min=1,
                            max=100,
                            step=1,
                            value=DEFAULT_CRF_MU,
                        ),

                        html.H6(id="crf-downsample-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="crf-downsample-slider",
                            min=1,
                            max=6,
                            step=1,
                            value=DEFAULT_CRF_DOWNSAMPLE,
                        ),

                        # html.H6(id="crf-gtprob-display"),
                        # # Slider for specifying pen width
                        # dcc.Slider(
                        #     id="crf-gtprob-slider",
                        #     min=0.5,
                        #     max=0.95,
                        #     step=0.05,
                        #     value=DEFAULT_CRF_GTPROB,
                        # ),

                        dcc.Markdown(
                            ">Classifier settings"
                        ),

                        html.H6(id="rf-downsample-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="rf-downsample-slider",
                            min=1,
                            max=20,
                            step=1,
                            value=DEFAULT_RF_DOWNSAMPLE,
                        ),

                        html.H6(id="numscales-display"),
                        # Slider for specifying pen width
                        dcc.Slider(
                            id="numscales-slider",
                            min=2,
                            max=6,
                            step=1,
                            value=DEFAULT_NUMSCALES,
                        ),

                    ],
                    className="three columns app-background",
                ),
            ],
            className="ten columns",
        ), #main content Div

        #========================================================
        ## tab 2
        #========================================================

        ]),
        dcc.Tab(label='File List and Instructions', children=[

        html.H4(children="Doodler"),
        dcc.Markdown(
            "> A user-interactive tool for fast segmentation of imagery (designed for natural environments), using a Multilayer Perceptron classifier and Conditional Random Field (CRF) refinement. \
            Doodles are used to make a classifier model, which maps image features to unary potentials to create an initial image segmentation. The segmentation is then refined using a CRF model."
        ),

            dcc.Input(id='my-id', value='Enter-user-ID', type="text"),
            html.Button('Submit', id='button'),
            html.Div(id='my-div'),

            html.H3("Select Image"),
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
            html.Div([html.Div(id='live-update-text'),
                      dcc.Interval(id='interval-component', interval=500, n_intervals=0)]),


        html.P(children="This image/Copy"),
        dcc.Textarea(id="thisimage_output", cols=80),
        html.Br(),

        dcc.Markdown(
            """
    **Instructions:**
    * Before you begin, make a new 'classes.txt' file that contains a list of the classes you'd like to label
    * Optionally, you can copy the images you wish to label into the 'assets' folder (just jpg, JPG or jpeg extension, or mixtures of those, for now)
    * Enter a user ID (initials or similar). This will get appended to your results to identify you. Results are also timestamped. You may enter a user ID at any time (or not at all)
    * Select an image from the list (often you need to select the image twice: make sure the image selected matches the image name shown in the box)
    * Make some brief annotations ('doodles') of every class present in the image, in every region of the image that class is present
    * Check 'Show/compute segmentation'. The computation time depends on image size, and the number of classes and doodles. Larger image or more doodles/classes = greater time and memory required
    * If you're not happy, uncheck 'Show/compute segmentation' and play with the parameters. However, it is often better to leave the parameters and correct mistakes by adding or removing doodles, or using a different pen width.
    * Once you're happy, you can download the label image, but it is already saved in the 'results' folder.
    * Before you move onto the next image from the list, uncheck 'Show/compute segmentation'.
    * Repeat. Happy doodling! Press Ctrl+C to end the program. Results are in the 'results' folder, timestamped. Session logs are also timestamped and found in the 'logs' directory.
    * As you go, the program only lists files that are yet to be labeled. It does this irrespective of your opinion of the segmentation, so you get 'one shot' before you select another image (i.e. you cant go back to redo)
    * [Code on GitHub](https://github.com/dbuscombe-usgs/dash_doodler).
    """
        ),
        dcc.Markdown(
            """
    **Tips:** 1) Works best for small imagery, typically much smaller than 3000 x 3000 px images. This prevents out-of-memory errors, and also helps you identify small features\
    2) Less is usually more! It is often best to use small pen width and relatively few annotations. Don't be tempted to spend too long doodling; extra doodles can be strategically added to correct segmentations \
    3) Make doodles of every class present in the image, and also every region of the image (i.e. avoid label clusters) \
    4) If things get weird, hit the refresh button on your browser and it should reset the application. Don't worry, all your previous work is saved!\
    5) Remember to uncheck 'Show/compute segmentation' before you change parameter values or change image\
    """
        ),

        ]),]),

        #========================================================
        ## components that are not displayed, used for storing data in localhost
        #========================================================

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


# ##========================================================
##========================================================
## app callbacks
##========================================================
@app.callback(
    [
    Output("select-image","options"),
    Output("graph", "figure"),
    Output("image-list-store", "data"),
    Output("masks", "data"),
    Output('my-div', 'children'),
    Output("segmentation", "data"),
    Output('thisimage_output', 'value'),
    Output("pen-width-display", "children"),
    Output("theta-display", "children"),
    Output("mu-display", "children"),
    Output("crf-downsample-display", "children"),
    # Output("crf-gtprob-display", "children"),
    Output("rf-downsample-display", "children"),
    Output("numscales-display", "children"),
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
    Input("crf-theta-slider", "value"),
    Input('crf-mu-slider', "value"),
    Input("pen-width", "value"),
    Input("crf-show-segmentation", "value"),
    Input("crf-downsample-slider", "value"),
    # Input("crf-gtprob-slider", "value"),
    Input("rf-downsample-slider", "value"),
    Input("numscales-slider", "value"),
    Input("select-image", "value"),
    Input('interval-component', 'n_intervals'),
    ],
    [
    State("image-list-store", "data"),
    State('my-id', 'value'),
    State("masks", "data"),
    State("segmentation", "data"),
    State("classified-image-store", "data"),
    ],
)

# ##========================================================
##========================================================
## app callback function
##========================================================
def update_output(
    uploaded_filenames,
    uploaded_file_contents,
    graph_relayoutData,
    any_label_class_button_value,
    crf_theta_slider_value,
    crf_mu_slider_value,
    pen_width_value,
    show_segmentation_value,
    crf_downsample_value,
    rf_downsample_value,
    n_sigmas,
    select_image_value,
    n_intervals,
    image_list_data,
    my_id_value,
    masks_data,
    segmentation_data,
    segmentation_store_data,
    ):
    """
    This is where all the action happens, and is called any time a button is pressed
    This function is automatically called, and the inputs and outputs match, in order,
    the list of callback inputs and outputs above

    The callback context is first defined, which dictates what the function does
    """

    callback_context = [p["prop_id"] for p in dash.callback_context.triggered][0]
    #print(callback_context)

    multichannel = True
    intensity = True
    edges = True
    texture = True

    image_list_data = []
    # all_image_value = ''
    files = ''
    options = []

    # Remove any "_" from my_id_value and if the my_id_value is empty replace with TEMPID
    my_id_value = my_id_value.replace("_","")
    if(len(my_id_value) == 0):
        my_id_value='TEMPID'

    if callback_context=='interval-component.n_intervals':
        #this file must exist - it contains a list of images labeled in this session
        filelist = 'files_done.txt'
        files, labeled_files = uploaded_files(filelist,UPLOAD_DIRECTORY,LABELED_DIRECTORY)

        logging.info('File list written to %s' % (filelist))

        files = [f.split('assets/')[-1] for f in files]
        labeled_files = [f.split('labeled/')[-1] for f in labeled_files]

        files = list(set(files) - set(labeled_files))
        files = sorted(files)

        options = [{'label': image, 'value': image } for image in files]

        logging.info('Checked assets and labeled lists and revised list of images yet to label')

    if select_image_value is not None:
        if 'assets' not in select_image_value:
            select_image_value = 'assets'+os.sep+select_image_value

    if callback_context == "graph.relayoutData":
        try:
            if "shapes" in graph_relayoutData.keys():
                masks_data["shapes"] = graph_relayoutData["shapes"]
            else:
                return dash.no_update
        except:
            return dash.no_update

    elif callback_context == "select-image.value":
       masks_data={"shapes": []}
       segmentation_data={}

       logging.info('New image selected')

    pen_width = pen_width_value

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
        stroke_color=convert_integer_class_to_color(class_label_colormap,label_class_value),
        pen_width=pen_width,
        shapes=masks_data["shapes"],
    )

    logging.info('Main figure window updated with new image')

    if ("Show segmentation" in show_segmentation_value) and (
        len(masks_data["shapes"]) > 0):
        # to store segmentation data in the store, we need to base64 encode the
        # PIL.Image and hash the set of shapes to use this as the key
        # to retrieve the segmentation data, we need to base64 decode to a PIL.Image
        # because this will give the dimensions of the image
        sh = shapes_to_key(
            [
                masks_data["shapes"],
                '', #segmentation_features_value,
                '', #sigma_range_slider_value,
            ]
        )

        segimgpng = None

        # start timer
        if os.name=='posix': # true if linux/mac or cygwin on windows
           start = time.time()
        else: # windows
           start = time.clock()

        # this is the function that computes and updates the segmentation whenever the checkbox is checked
        segimgpng, seg, img, color_doodles, doodles  = show_segmentation(
            [select_image_value], masks_data["shapes"], callback_context,
             crf_theta_slider_value, crf_mu_slider_value, results_folder, rf_downsample_value, crf_downsample_value, 1.0, my_id_value,
             n_sigmas, multichannel, intensity, edges, texture,class_label_colormap
        )

        logging.info('... showing segmentation on screen')
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        if os.name=='posix': # true if linux/mac
           elapsed = (time.time() - start)#/60
        else: # windows
           elapsed = (time.clock() - start)#/60

        logging.info('Processing took %s seconds' % (str(elapsed)))

        lstack = (np.arange(seg.max()) == seg[...,None]-1).astype(int) #one-hot encode the 2D label into 3D stack of IxJxN classes

        logging.info('One-hot encoded label stack created')


        if type(select_image_value) is list:
            if 'jpg' in select_image_value[0]:
                colfile = select_image_value[0].replace('assets',results_folder).replace('.jpg','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')
            if 'JPG' in select_image_value[0]:
                colfile = select_image_value[0].replace('assets',results_folder).replace('.JPG','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')
            if 'jpeg' in select_image_value[0]:
                colfile = select_image_value[0].replace('assets',results_folder).replace('.jpeg','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')

            if np.ndim(img)==3:
                imsave(colfile,label_to_colors(seg-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
            else:
                imsave(colfile,label_to_colors(seg-1, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

            orig_image = imread(select_image_value[0])
            if np.ndim(orig_image)>3:
               orig_image = orig_image[:,:,:3]

        else:
            if 'jpg' in select_image_value:
                colfile = select_image_value.replace('assets',results_folder).replace('.jpg','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')
            if 'JPG' in select_image_value:
                colfile = select_image_value.replace('assets',results_folder).replace('.JPG','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')
            if 'jpeg' in select_image_value:
                colfile = select_image_value.replace('assets',results_folder).replace('.jpeg','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')

            if np.ndim(img)==3:
                imsave(colfile,label_to_colors(seg-1, img[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))
            else:
                imsave(colfile,label_to_colors(seg-1, img==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False))

            orig_image = imread(select_image_value)
            if np.ndim(orig_image)>3:
               orig_image = orig_image[:,:,:3]
            # orig_image = img_to_ubyte_array(select_image_value)

        logging.info('RGB label image saved to %s' % (colfile))

        settings_dict = np.array([pen_width, crf_downsample_value, rf_downsample_value, crf_theta_slider_value, crf_mu_slider_value, 1.0, n_sigmas])

        if type(select_image_value) is list:
            if 'jpg' in select_image_value[0]:
                numpyfile = select_image_value[0].replace('assets',results_folder).replace('.jpg','_'+my_id_value+'.npz')
            if 'JPG' in select_image_value[0]:
                numpyfile = select_image_value[0].replace('assets',results_folder).replace('.JPG','_'+my_id_value+'.npz')
            if 'jpeg' in select_image_value[0]:
                numpyfile = select_image_value[0].replace('assets',results_folder).replace('.jpeg','_'+my_id_value+'.npz')

            if os.path.exists(numpyfile):
                saved_data = np.load(numpyfile)
                savez_dict = dict()
                for k in saved_data.keys():
                    tmp = saved_data[k]
                    name = str(k)
                    savez_dict['0'+name] = tmp
                    del tmp

                savez_dict['image'] = img.astype(np.uint8)
                savez_dict['orig_image'] = orig_image.astype(np.uint8)
                savez_dict['label'] = lstack.astype(np.uint8)
                savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                savez_dict['doodles'] = doodles.astype(np.uint8)
                savez_dict['settings'] = settings_dict
                savez_dict['classes'] = class_label_names
                np.savez_compressed(numpyfile, **savez_dict )

            else:
                savez_dict = dict()
                savez_dict['image'] = img.astype(np.uint8)
                savez_dict['orig_image'] = orig_image.astype(np.uint8)
                savez_dict['label'] = lstack.astype(np.uint8)
                savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                savez_dict['doodles'] = doodles.astype(np.uint8)
                savez_dict['settings'] = settings_dict
                savez_dict['classes'] = class_label_names
                np.savez_compressed(numpyfile, **savez_dict ) #save settings too

        else:
            if 'jpg' in select_image_value:
                numpyfile = select_image_value.replace('assets',results_folder).replace('.jpg','_'+my_id_value+'.npz')
            if 'JPG' in select_image_value:
                numpyfile = select_image_value.replace('assets',results_folder).replace('.JPG','_'+my_id_value+'.npz')
            if 'jpeg' in select_image_value:
                numpyfile = select_image_value.replace('assets',results_folder).replace('.jpeg','_'+my_id_value+'.npz')

            if os.path.exists(numpyfile):
                saved_data = np.load(numpyfile)
                savez_dict = dict()
                for k in saved_data.keys():
                    tmp = saved_data[k]
                    name = str(k)
                    savez_dict['0'+name] = tmp
                    del tmp

                savez_dict['image'] = img.astype(np.uint8)
                savez_dict['orig_image'] = orig_image.astype(np.uint8)
                savez_dict['label'] = lstack.astype(np.uint8)
                savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                savez_dict['doodles'] = doodles.astype(np.uint8)
                savez_dict['settings'] = settings_dict
                savez_dict['classes'] = class_label_names
                np.savez_compressed(numpyfile, **savez_dict )#save settings too

            else:
                savez_dict = dict()
                savez_dict['image'] = img.astype(np.uint8)
                savez_dict['orig_image'] = orig_image.astype(np.uint8)
                savez_dict['label'] = lstack.astype(np.uint8)
                savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                savez_dict['doodles'] = doodles.astype(np.uint8)
                savez_dict['settings'] = settings_dict
                savez_dict['classes'] = class_label_names
                np.savez_compressed(numpyfile, **savez_dict )#save settings too

        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        del img, seg, lstack, doodles, color_doodles
        logging.info('Numpy arrays saved to %s' % (numpyfile))

        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

        segmentation_data = shapes_seg_pair_as_dict(
            segmentation_data, sh, segimgpng
        )
        try:
            segmentation_store_data = pil2uri(
                seg_pil(
                    select_image_value, segimgpng, do_alpha=True
                ) #plot_utils.
            )
            shutil.copyfile(select_image_value, select_image_value.replace('assets', 'labeled')) #move
        except:
            segmentation_store_data = pil2uri(
                seg_pil(
                    PIL.Image.open(select_image_value), segimgpng, do_alpha=True
                ) #plot_utils.
            )
            shutil.copyfile(select_image_value, select_image_value.replace('assets', 'labeled')) #move

        logging.info('%s moved to labeled folder' % (select_image_value.replace('assets', 'labeled')))

        images_to_draw = []
        if segimgpng is not None:
            images_to_draw = [segimgpng]

        fig = add_layout_images_to_fig(fig, images_to_draw) #plot_utils.

        show_segmentation_value = []
        image_list_data.append(select_image_value)

        try:
          os.remove('my_defaults.py')
        except:
          pass

        #write defaults back out to file
        with open('my_defaults.py', 'a') as the_file:
            the_file.write('DEFAULT_PEN_WIDTH = {}\n'.format(pen_width))
            the_file.write('DEFAULT_CRF_DOWNSAMPLE = {}\n'.format(crf_downsample_value))
            the_file.write('DEFAULT_RF_DOWNSAMPLE = {}\n'.format(rf_downsample_value))
            the_file.write('DEFAULT_CRF_THETA = {}\n'.format(crf_theta_slider_value))
            the_file.write('DEFAULT_CRF_MU = {}\n'.format(crf_mu_slider_value))
            # the_file.write('DEFAULT_CRF_GTPROB = {}\n'.format(gt_prob))
            the_file.write('DEFAULT_NUMSCALES = {}\n'.format(n_sigmas))

        print('my_defaults.py overwritten with parameter settings')

        # logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('my_defaults.py overwritten with parameter settings')

        # logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.info('percent RAM usage: %f' % (psutil.virtual_memory()[2]))

    if len(files) == 0:
        return [
        options,
        fig,
        image_list_data,
        masks_data,
        segmentation_data,
        'User ID: "{}"'.format(my_id_value) ,
        select_image_value,
        "Pen width (default: %d): %d" % (DEFAULT_PEN_WIDTH,pen_width),
        "Blur factor (default: %d): %d" % (DEFAULT_CRF_THETA, crf_theta_slider_value),
        "Model independence factor (default: %d): %d" % (DEFAULT_CRF_MU,crf_mu_slider_value),
        "CRF downsample factor (default: %d): %d" % (DEFAULT_CRF_DOWNSAMPLE,crf_downsample_value),
        # "User-defined quality score (1=perfect. default: %f): %f" % (DEFAULT_CRF_GTPROB,gt_prob),
        "Classifier downsample factor (default: %d): %d" % (DEFAULT_RF_DOWNSAMPLE,rf_downsample_value),
        "Number of scales (default: %d): %d" % (DEFAULT_NUMSCALES,n_sigmas),
        segmentation_store_data,
        ]
    else:
        return [
        options,
        fig,
        image_list_data,
        masks_data,
        segmentation_data,
        'User ID: "{}"'.format(my_id_value) ,
        select_image_value,
        "Pen width (default: %d): %d" % (DEFAULT_PEN_WIDTH,pen_width),
        "Blur factor (default: %d): %d" % (DEFAULT_CRF_THETA, crf_theta_slider_value),
        "Model independence factor  (default: %d): %d" % (DEFAULT_CRF_MU,crf_mu_slider_value),
        "CRF downsample factor (default: %d): %d" % (DEFAULT_CRF_DOWNSAMPLE,crf_downsample_value),
        # "User-defined quality score (1=perfect. default: %f): %f" % (DEFAULT_CRF_GTPROB,gt_prob),
        "Classifier downsample factor (default: %d): %d" % (DEFAULT_RF_DOWNSAMPLE,rf_downsample_value),
        "Number of scales (default: %d): %d" % (DEFAULT_NUMSCALES,n_sigmas),
        segmentation_store_data,
        ]
