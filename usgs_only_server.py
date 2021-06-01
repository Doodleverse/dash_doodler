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


########################################################
############ IMPORTS ############################################
########################################################

# ##========================================================
# allows loading of functions from the src directory
import sys
sys.path.insert(1, 'src')

##========================================================
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

# pip install dash-auth
import dash_auth

from annotations_to_segmentations import *
from plot_utils import *
import fsspec
import io, base64, PIL.Image, json, shutil, os, time, subprocess
from glob import glob
from datetime import datetime
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory

##========================================================
import logging
logging.basicConfig(filename=os.getcwd()+os.sep+'logs/'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'.log',  level=logging.INFO) #DEBUG) #encoding='utf-8',
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


########################################################
############ SETTINGS / FILES ############################################
########################################################


S3_DATABUCKET = 's3://cmgp-upload-download-bucket/watermasker1/'
S3_RESULTSBUCKET = 's3://cmgp-upload-download-bucket/watermasker1/results'

#========================================================
fs = fsspec.filesystem('s3', profile='default')
# replace bucketname with cmd input
s3files = fs.ls(S3_DATABUCKET)
s3files = [f for f in s3files if 'jpg' in f]
Ns3files = len(s3files)
print("%i files in s3 bucket" % (Ns3files))

fs_res = fsspec.filesystem('s3', profile='default')
# replace bucketname with cmd input
resfiles = fs_res.ls(S3_RESULTSBUCKET)
Nresfiles = len(resfiles)
print("%i results files in s3 bucket" % (Nresfiles))


##========================================================
DEFAULT_IMAGE_PATH = "assets/logos/dash-default.jpg"

DEFAULT_PEN_WIDTH = 3
DEFAULT_CRF_DOWNSAMPLE = 4
DEFAULT_RF_DOWNSAMPLE = 8
DEFAULT_CRF_THETA = 1
DEFAULT_CRF_MU = 1
DEFAULT_RF_NESTIMATORS = 3
DEFAULT_CRF_GTPROB = 0.9

# the number of different classes for labels
DEFAULT_LABEL_CLASS = 0

UPLOAD_DIRECTORY = os.getcwd()+os.sep+"assets"
LABELED_DIRECTORY = os.getcwd()+os.sep+"labeled"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

##========================================================

try:
    with open('classes.txt') as f:
        classes = f.readlines()
except: #in case classes.txt does not exist
    print("classes.txt not found or badly formatted. Exit the program and fix the classes.txt file ... otherwie, will continue using default classes. ")
    classes = ['water', 'land']

class_label_names = [c.strip() for c in classes]

NUM_LABEL_CLASSES = len(class_label_names)

if NUM_LABEL_CLASSES<=10:
    class_label_colormap = px.colors.qualitative.G10
else:
    class_label_colormap = px.colors.qualitative.Light24

# we can't have fewer colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

class_labels = list(range(NUM_LABEL_CLASSES))

logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
logging.info('loaded class labels:')
for f in class_label_names:
    logging.info(f)

rf_file = 'RandomForestClassifier_'+'_'.join(class_label_names)+'.pkl.z'    #class_label_names
data_file = 'data_'+'_'.join(class_label_names)+'.pkl.z'    #class_label_names

try:
    shutil.move(rf_file, rf_file.replace('.pkl.z','_'+datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+'.pkl.z'))
except:
    pass


try:
    shutil.move(data_file, data_file.replace('.pkl.z','_'+datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+'.pkl.z'))
except:
    pass


##========================================================
results_folder = 'results/results'+datetime.now().strftime("%Y-%m-%d-%H-%M")

try:
    os.mkdir(results_folder)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info("Folder created: %s" % (results_folder))
except:
    pass

logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
logging.info("Results will be written to %s" % (results_folder))


## while file not in assets/ ...
usefile = np.random.randint(Ns3files)
file = s3files[usefile]
# print(file.split(os.sep)[-1])
fp = 's3://'+file
with fs.open(fp, 'rb') as f:
    img = np.array(PIL.Image.open(f))[:,:,:3]
    f.close()
    imsave('assets/'+file.split(os.sep)[-1], img)

# downloads 1 image

files = sorted(glob('assets/*.jpg')) + sorted(glob('assets/*.JPG')) + sorted(glob('assets/*.jpeg'))

files = [f for f in files if 'dash' not in f]

logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
logging.info('loaded files:')
for f in files:
    logging.info(f)


########################################################
############ FUNCTIONS ############################################
########################################################

##========================================================
def convert_integer_class_to_color(n):
    return class_label_colormap[n]

def convert_color_class(c):
    return class_label_colormap.index(c)

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
            "height": 650
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
            if 'JPG' in filename:
                files.append(filename)
            if 'jpeg' in filename:
                files.append(filename)

    labeled_files = []
    for filename in os.listdir(LABELED_DIRECTORY):
        path = os.path.join(LABELED_DIRECTORY, filename)
        if os.path.isfile(path):
            if 'jpg' in filename:
                labeled_files.append(filename)
            if 'JPG' in filename:
                labeled_files.append(filename)
            if 'jpeg' in filename:
                labeled_files.append(filename)

    filelist = 'files_done.txt'

    with open(filelist, 'w') as filehandle:
        for listitem in labeled_files:
            filehandle.write('%s\n' % listitem)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('File list written to %s' % (filelist))

    return sorted(files), sorted(labeled_files)


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


##========================================================
def show_segmentation(image_path,
    mask_shapes,
    callback_context,
    crf_theta_slider_value,
    crf_mu_slider_value,
    results_folder,
    rf_downsample_value,
    crf_downsample_factor,
    my_id_value,
    rf_file,
    data_file,
    multichannel,
    intensity,
    edges,
    texture,
    ):

    gt_prob = .9
    n_estimators = 3

    """ adds an image showing segmentations to a figure's layout """

    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [convert_color_class(shape["line"]["color"]) + 1 for shape in mask_shapes]

    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }

    sigma_min=1; sigma_max=16

    segimg, seg, img, color_doodles, doodles = compute_segmentations(
        mask_shapes, crf_theta_slider_value,crf_mu_slider_value,
        results_folder, rf_downsample_value, # median_filter_value,
        crf_downsample_factor, gt_prob, my_id_value, callback_context, rf_file, data_file,
        multichannel, intensity, edges, texture, 1, 16, n_estimators,
        img_path=image_path,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
    )

    # get the classifier that we can later store in the Store
    segimgpng = img_array_2_pil(segimg) #plot_utils.

    return (segimgpng, seg, img, color_doodles, doodles )


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

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))

##===============================================================

## remove

UPLOAD_DIRECTORY = os.getcwd()+os.sep+"assets"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    logging.info('Made the directory '+UPLOAD_DIRECTORY)

##========================================================

server = Flask(__name__)
app = dash.Dash(server=server)

try:
    with open('users/users.json') as f:
        VALID_USERNAME_PASSWORD_PAIRS = json.load(f)

    #!pip install dash-auth
    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )
except:
    print("Credentials not found or badly formatted..does users/users.json exist? Is there a trailing comma?")

##========================================================

########################################################
############ BUILD APP ############################################
########################################################

app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id="banner",
            children=[
                html.H2(
            "Doodler: Fast Interactive Segmentation of Imagery",
            id="title",
            className="seven columns",
        ),
        html.Img(id="logo", src=app.get_asset_url("logos/dash-logo-new.png")),
        # html.Div(html.Img(src=app.get_asset_url('logos/dash-logo-new.png'), style={'height':'10%', 'width':'10%'})), #id="logo",

        html.H2(""),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["                                 Label all classes that are present, in all regions of the image those classes occur."]
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
                                    figure=make_and_return_default_figure(),
                                    config={
                                        'displayModeBar': 'hover',
                                        "displaylogo": False,
                                        # 'modeBarOrientation': 'h',
                                        "modeBarButtonsToAdd": [
                                            "drawrect",
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

                dcc.Input(id='my-id', value='Enter-user-ID', type="text"),
                html.Button('Submit', id='button'),
                html.Div(id='my-div'),

                # html.H3("Select Image"),
                dcc.Dropdown(
                    id="select-image",
                    optionHeight=15,
                    style={'display': 'none'},#{'fontSize': 13},
                    options = [
                        {'label': image.split('assets/')[-1], 'value': image } \
                        for image in files
                    ],

                    value='assets/logos/dash-default.jpg', #
                    multi=False,
                ),

                dcc.Textarea(id="thisimage_output", cols=80, style={'display': 'none'}),

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
                            max=5,
                            step=1,
                            value=DEFAULT_PEN_WIDTH,
                        ),


                        # html.Button('Submit', id='submitbutton'),

                        # Indicate showing most recently computed segmentation
                        dcc.Checklist(
                            id="crf-show-segmentation",
                            options=[
                                {
                                    "label": "SEGMENT IMAGE",
                                    "value": "Show segmentation",
                                }
                            ],
                            value=[],
                        ),

                    ],
                    className="three columns app-background",
                ),
            ],
            className="ten columns",
        ), #main content Div
        ]),


        ]),

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


# ##========================================================
########################################################
############ UPDATE OUTPUT FUNCTION CALLBACKS ############################################
########################################################

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
    Input("crf-show-segmentation", "value"),
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
########################################################
############ UPDATE OUTPUT FUNCTION ############################################
########################################################

def update_output(
    uploaded_filenames,
    uploaded_file_contents,
    graph_relayoutData,
    any_label_class_button_value,
    pen_width_value,
    show_segmentation_value,
    image_list_data,
    my_id_value,
    masks_data,
    segmentation_data,
    segmentation_store_data,
    ):
    """Save uploaded files and regenerate the file list."""

    callback_context = [p["prop_id"] for p in dash.callback_context.triggered][0]
    print(callback_context)

    multichannel = True
    intensity = True
    edges = True
    texture = True
    crf_theta_slider_value = 1
    crf_mu_slider_value = 1
    rf_downsample_value = 8
    crf_downsample_value = 4

    image_list_data = []
    all_image_value = ''
    files = ''
    options = []

    # if callback_context=='interval-component.n_intervals':
    files, labeled_files = uploaded_files()

    files = [f.split('assets/')[-1] for f in files]
    labeled_files = [f.split('labeled/')[-1] for f in labeled_files]

    files = list(set(files) - set(labeled_files))
    files = sorted(files)

    options = [{'label': image, 'value': image } for image in files]

    # print(files)

    if len(files)>0:
        select_image_value = files[0]
    else:
        print("No more files")


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


    pen_width = pen_width_value #int(round(2 ** (pen_width_value)))

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
                '', #segmentation_features_value,
                '', #sigma_range_slider_value,
            ]
        )

        rf_file = 'RandomForestClassifier_'+'_'.join(class_label_names)+'.pkl.z'    #class_label_names
        data_file = 'data_'+'_'.join(class_label_names)+'.pkl.z'    #class_label_names

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Saving RF model to %s' % (rf_file))

        logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        logging.info('Saving data features to %s' % (rf_file))

        segimgpng = None
        if 'median' not  in callback_context:

            # start timer
            if os.name=='posix': # true if linux/mac or cygwin on windows
               start = time.time()
            else: # windows
               start = time.clock()

            segimgpng, seg, img, color_doodles, doodles  = show_segmentation(
                [select_image_value], masks_data["shapes"], callback_context,#median_filter_value,
                 crf_theta_slider_value, crf_mu_slider_value, results_folder, rf_downsample_value, crf_downsample_value, my_id_value, rf_file, data_file, #gt_prob,
                 multichannel, intensity, edges, texture,#n_estimators, # sigma_range_slider_value[0], sigma_range_slider_value[1],
            )

            if os.name=='posix': # true if linux/mac
               elapsed = (time.time() - start)/60
            else: # windows
               elapsed = (time.clock() - start)/60
            #print("Processing took "+ str(elapsed) + " minutes")

            lstack = (np.arange(seg.max()) == seg[...,None]-1).astype(int) #one-hot encode

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

            else:
                #colfile = select_image_value.replace('assets',results_folder).replace('.jpg','_label'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'_'+my_id_value+'.png')
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

            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('RGB label image saved to %s' % (colfile))

            gt_prob = .9
            n_estimators = 3
            settings_dict = np.array([pen_width, crf_downsample_value, rf_downsample_value, crf_theta_slider_value, crf_mu_slider_value,  n_estimators, gt_prob])#median_filter_value,sigma_range_slider_value[0], sigma_range_slider_value[1]

            if type(select_image_value) is list:
                if 'jpg' in select_image_value[0]:
                    numpyfile = select_image_value[0].replace('assets',results_folder).replace('.jpg','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+
                if 'JPG' in select_image_value[0]:
                    numpyfile = select_image_value[0].replace('assets',results_folder).replace('.JPG','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+
                if 'jpeg' in select_image_value[0]:
                    numpyfile = select_image_value[0].replace('assets',results_folder).replace('.jpeg','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+


                if os.path.exists(numpyfile):
                    saved_data = np.load(numpyfile)
                    savez_dict = dict()
                    for k in saved_data.keys():
                        tmp = saved_data[k]
                        name = str(k)
                        savez_dict['0'+name] = tmp
                        del tmp

                    savez_dict['image'] = img.astype(np.uint8)
                    savez_dict['label'] = lstack.astype(np.uint8)
                    savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                    savez_dict['doodles'] = doodles.astype(np.uint8)
                    savez_dict['settings'] = settings_dict
                    np.savez(numpyfile, **savez_dict )

                else:
                    savez_dict = dict()
                    savez_dict['image'] = img.astype(np.uint8)
                    savez_dict['label'] = lstack.astype(np.uint8)
                    savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                    savez_dict['doodles'] = doodles.astype(np.uint8)
                    savez_dict['settings'] = settings_dict

                    np.savez(numpyfile, **savez_dict ) #save settings too

            else:
                if 'jpg' in select_image_value:
                    numpyfile = select_image_value.replace('assets',results_folder).replace('.jpg','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+
                if 'JPG' in select_image_value:
                    numpyfile = select_image_value.replace('assets',results_folder).replace('.JPG','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+
                if 'jpeg' in select_image_value:
                    numpyfile = select_image_value.replace('assets',results_folder).replace('.jpeg','_'+my_id_value+'.npz') #datetime.now().strftime("%Y-%m-%d-%H-%M")+

                if os.path.exists(numpyfile):
                    saved_data = np.load(numpyfile)
                    savez_dict = dict()
                    for k in saved_data.keys():
                        tmp = saved_data[k]
                        name = str(k)
                        savez_dict['0'+name] = tmp
                        del tmp

                    savez_dict['image'] = img.astype(np.uint8)
                    savez_dict['label'] = lstack.astype(np.uint8)
                    savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                    savez_dict['doodles'] = doodles.astype(np.uint8)
                    savez_dict['settings'] = settings_dict

                    np.savez(numpyfile, **savez_dict )#save settings too

                else:
                    savez_dict = dict()
                    savez_dict['image'] = img.astype(np.uint8)
                    savez_dict['label'] = lstack.astype(np.uint8)
                    savez_dict['color_doodles'] = color_doodles.astype(np.uint8)
                    savez_dict['doodles'] = doodles.astype(np.uint8)
                    savez_dict['settings'] = settings_dict

                    np.savez(numpyfile, **savez_dict )#save settings too

            del img, seg, lstack, doodles, color_doodles
            logging.info(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            logging.info('Numpy arrays saved to %s' % (numpyfile))

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

        images_to_draw = []
        if segimgpng is not None:
            images_to_draw = [segimgpng]

        fig = add_layout_images_to_fig(fig, images_to_draw) #plot_utils.

        image_list_data.append(select_image_value)

        masks_data={"shapes": []}
        segmentation_data={}

        # grab a new random file
        ## while file not in assets/ ...
        usefile = np.random.randint(Ns3files)
        file = s3files[usefile]
        fp = 's3://'+file
        with fs.open(fp, 'rb') as f:
            img = np.array(PIL.Image.open(f))[:,:,:3]
            f.close()
            imsave('assets/'+file.split(os.sep)[-1], img)

        to_write = numpyfile.split(results_folder)[-1].split(os.sep)[-1]
        # print(to_write)
        #subprocess.run(["aws","s3","cp", numpyfile, S3_RESULTSBUCKET])
        #os.system("aws s3 cp "+numpyfile+" "+S3_RESULTSBUCKET)
        # subprocess.Popen(["/usr/bin/aws","s3","cp", numpyfile, S3_RESULTSBUCKET+'/'+to_write])
        subprocess.Popen(["aws","s3","cp", numpyfile, S3_RESULTSBUCKET+'/'+to_write])


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
        segmentation_store_data,
        ]


##========================================================

########################################################
############ RUN APP ############################################
########################################################

if __name__ == "__main__":
    print('This is for certain USGS personnel only, and will not work for you unless you have specific AWS keys')
    # print('Go to http://127.0.0.1:8050/ in your web browser to use Doodler')
    app.run_server()
