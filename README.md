# 📦 Dash-Doodler

[![Last Commit](https://img.shields.io/github/last-commit/dbuscombe-usgs/dash_doodler)](
https://github.com/dbuscombe-usgs/dash_doodler/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dbuscombe-usgs/dash_doodler/graphs/commit-activity)
[![Wiki](https://img.shields.io/badge/wiki-documentation-forestgreen)](https://github.com/dbuscombe-usgs/dash_doodler/wiki)
![GitHub](https://img.shields.io/github/license/dbuscombe-usgs/dash_doodler)
[![Wiki](https://img.shields.io/badge/discussion-active-forestgreen)](https://github.com/dbuscombe-usgs/dash_doodler/discussions)

![Doodler Logo](./doodler-logo.png)

## 🌟 Highlights
This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation. The video shows a basic usage of doodler. 1) Annotate the scene with a few examples of each class (colorful buttons).  2) Check 'compute and show segmentation' and wait for the result. The label image is written to the 'results' folder

Here's a movie of Doodler in action:

![](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/short_1024px_30fps.gif)

## ℹ️ Documentation

### Website
Check out the [Doodler website](https://dbuscombe-usgs.github.io/dash_doodler/)

### Paper
[![Earth ArXiv Preprint
DOI](https://img.shields.io/badge/%F0%9F%8C%8D%F0%9F%8C%8F%F0%9F%8C%8E%20EarthArXiv-doi.org%2F10.31223%2FX59K83-%23FF7F2A)](https://doi.org/10.31223/X59K83)

### Code that made the paper
[![DOI](https://zenodo.org/badge/304798940.svg)](https://zenodo.org/badge/latestdoi/304798940)

### Data that made the paper
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.2fqz612ps)

### Overview
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

What is generally required in the above case is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image. The number of pixels annotated by the human annotator is typically a small fraction of the total pixels in the image.  

`Doodler` is a tool for sparse, not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Multilayer Perceptron model for initial predictions, refined by a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more. Your annotations are first used to train and apply a random forest on the entire image, then a CRF is used to refine labels further based on the underlying image.

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, create a `classes.txt` file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. The maximum number of classes allowed is 24. The images that you upload will go into the `assets/` folder. The labels images you create are written to the `results` folder.

## ✍️ Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs) Marda Science / USGS Pacific Coastal and Marine Science Center. Developed originally for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

Contributions:
* [@2320sharon](https://github.com/2320sharon)
* [@ebgoldstein](https://github.com/ebgoldstein)

### <a name="ack"></a>Acknowledgements
Doodler is based on code previously contained in the "doodle_labeller" [repository](https://github.com/dbuscombe-usgs/doodle_labeller) which implements a similar algorithm in OpenCV. The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244). Inspired by [this plotly example](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation) and the previous openCV based implementation [doodle_labeller](https://github.com/dbuscombe-usgs/doodle_labeller), that actually has origins in a USGS CDI-sponsored class I taught in summer of 2018, called [dl-tools](https://github.com/dbuscombe-usgs/dl_tools). So, it's been a 3+ year effort!


## ⬇️ Installation

> Check out the installation guide on the [Doodler website](https://dbuscombe-usgs.github.io/dash_doodler/docs/tutorial-basics/deploy-local)


We advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `dashdoodler`

```
conda env create --file install/dashdoodler-clean.yml
conda activate dashdoodler
```

*If* the above doesn't work, try this:

```bash
conda env create --file environment/dashdoodler.yml
conda activate dashdoodler
```

*If neither of the above* work, try this:

```bash
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r environment/requirements.txt
```

and good luck to you!

## 🚀 Usage

> Check out the user guide on the [Doodler website](https://dbuscombe-usgs.github.io/dash_doodler/docs/tutorial-basics/what-to-do)

Move your images into the `assets` folder. For the moment, they must be jpegs with the `.jpg` (or `JPG` or `jpeg`) extension. Support for other image types forthcoming ...

Run the app. An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Tested so far with Chrome, Firefox, and Edge.

```bash
python doodler.py
```

Open a browser and go to 127.0.0.1:8050. You may have to hit the refresh button. If, after some time doodling things seem odd or buggy, sometimes a browser refresh will fix those glitches.

### Example screenshots of use with example dataset

(note: these are screengrabs of an older version of the program, so the buttons and their names are now slightly different)

#### `doodler.py`

![Example 1](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py.png)
![Example 2](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py1.png)
![Example 3](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py2.png)
![Example 4](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py3.png)
![Example 5](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py4.png)
![Example 6](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py5.png)

### Videos
More demonstration videos (older version of the program):

![Doodler example 2](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/quick-saturban-x2c.gif)

![Doodler example 3](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short.gif)

![Elwha example](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-elwha.gif)

![Coast Train example](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-coast.gif)

![Coast Train example 2](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-coast2.gif)


### <a name="coasttrain"></a>Unpacking Coast Train Data

To use the labels in their native class sets (that vary per image), use the `gen_images_and_labels_4_zoo.py` script as described below. To use the labels in remapped classes (standardized across image sets), use the `gen_remapped_images_and_labels.py` script described below.


### <a name="utilities"></a>Utility scripts

Doodler is compatible with the partner segmentation program, [Zoo](https://github.com/dbuscombe-usgs/segmentation_zoo) in a couple of different ways:

1. You could run the function `gen_npz_4_zoo.py` to create npz files that contain only image and label pairs. This is the same output as you would get from running the Zoo program `make_nd_datasets.py'

2. You could alternatively run the function `gen_images_and_labels_4_zoo.py` that would generate jpeg greyscale image files and label image jpegs for use with the Zoo program `make_nd_datasets.py'.

3. Finally, you could run the function `gen_remapped_images_and_labels.py` that would generate jpeg greyscale image files and remapped label image jpegs for use with the Zoo program `make_nd_datasets.py'. Labels are remapped based on a dictionary of class aliases and a list of classes present, using a special config file. To remap Coast Train data, use the config files provided [here](https://github.com/dbuscombe-usgs/CoastTrainMetaPlots/tree/main/remap_config_files) 

The first scenario might be most common because it requires one less step, however the second scenario might be useful for using the labels with another software package, or for further post-processing of the labels

There are two additional scripts in the `utils` folder:

1. `viz_npz.py` creates transparent overlay plots of images and labels, and has three modes with the following syntax `viz_npz.py [-t npz type {0}/1/2]` where optional `-t` controls what type of npz file: native from doodler (option 0, default), a `labelgen` file from `plot_label_generation.py`, a npz file used as input for Zoo

2. `plot_label_generation.py` that generates a detailed sequence of plots for every input npz file from doodler, including plots of the doodles themselves, overlays, and internal model outputs.


## 💭 Feedback and Contributing

Please read our [code of conduct](https://github.com/dbuscombe-usgs/dash_doodler/blob/main/CODE_OF_CONDUCT.md)

Please contribute to the [Discussions tab](https://github.com/dbuscombe-usgs/dash_doodler/discussions) - we welcome your ideas and feedback.

We also invite all to open issues for bugs/feature requests using the [Issues tab](https://github.com/dbuscombe-usgs/dash_doodler/issues)


### <a name="contribute"></a>Contributing
Contributions are welcome, and they are greatly appreciated! Credit will always be given.

#### Report Bugs

Report bugs at https://github.com/dbuscombe-usgs/dash_doodler/issues.

Please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.
    * the log file made by the program during the session, found in

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### Write Documentation

We could always use more documentation, whether as part of the docs, in docstrings, or using this software in blog posts, articles, etc.

#### Get Started!

> See the [how to contribute](https://dbuscombe-usgs.github.io/dash_doodler/docs/tutorial-extras/how-to-contribute) section of the Doodler website

Ready to contribute? Here's how to set up for local development.

* Fork the dash_doodler repo on GitHub.

* Clone your fork locally:

`$ git clone git@github.com:your_name_here/dash_doodler.git`

Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

`$ cd dash_doodler/`
`$ conda env create --file install/dashdoodler.yml`
`$ conda activate dashdoodler`

Create a branch for local development:

`$ git checkout -b name-of-your-bugfix-or-feature`

Now you can make your changes locally.

Commit your changes and push your branch to GitHub:

`$ git add .`

`$ git commit -m "Your detailed description of your changes."`

`$ git push origin name-of-your-bugfix-or-feature`

Submit a pull request through the GitHub website.


## <a name="developers"></a>Developers notes

### Entrypoint
* The entrypoint is `doodler.py`, which will first download sample imagery if `DOWNLOAD_SAMPLE=True` in `environment\settings.py`.

* By default, `DOWNLOAD_SAMPLE=False` so imagery is not downloaded.

* The other variables in `environment\settings.py` are found in Dash's `app.run_server()` documentation.
  * `HOST="127.0.0.1"`` (should be `#"0.0.0.0"` for web deployment)
  * `PORT="8050"`
  * `DEBUG=False`
  * `DEV_TOOLS_PROPS_CHECK=False`

* `doodler.py` basically just calls and serves `app`, from `app.py`

### Application

* Loads classes and files and creates results folders and log file
* Creates the application layout and links all buttons to callback functions

### Callbacks
* utility functions are in `app_files\src\app_funcs.py`
* functions for drawing the imagery on the screen and making label overlays are in `app_files\src\plot_utils.py`
* functions for converting SVG annotations to raster label annotations and segmentations are in `app_files\src\annotations_to_segmentations.py`
* image segmentation/ML functions are in `app_files\src\image_segmentation.py`


<!--
```
sudo docker volume create doodler_data
sudo docker run -p 8050:8050 mardascience/dash_doodler:d1
sudo docker volume inspect doodler_data

sudo docker volume create --driver local -o o=bind -o type=none -o device="/home/marda/test" doodler_data

sudo docker run -d -p 8050:8050 --name doodler_container --mount source=doodler_data,target=/app  mardascience/dash_doodler:d1
sudo docker inspect doodler_container
```-->

### Docker workflows

To build your own docker image based on miniconda `continuumio/miniconda3`, called `doodler_docker_image`:


```
docker build -t doodler_docker_image .
```

then when it has finished building (it takes a while), check its size

```
sudo docker image ls doodler_docker_image
```

It is large - 4.8 GB. Run it in a container called `www`:

```
sudo docker run -p 8050:8050 -d -it --name www doodler_docker_image
```

The terminal will show no output, but you can see the process running a few different ways

Lists running containers:

```
docker ps
```

the container name will be at the end of the line of output of docker ps (images don't have logs; they're like classes)
```
docker logs [container_name] -
```


To stop and remove:

```
sudo docker stop www
sudo docker rm www
```

Please don't ask me about Docker - that's all I know. Please contribute Docker workflows and suggestions!


### <a name="progress"></a>Progress report

10/20/20:
* display numbers for every parameter
* fixed label creation and display in situations where there is a null image value (0), and where there are not as many classes in the scene as in the collection

10/22/20
* modified layout so image window is larger, and button bank is narrower. Hopefully easier to label, less zooming, etc. see https://github.com/dbuscombe-usgs/dash_doodler/issues/4. Thanks Dan Nowacki
* added yml installation file, modified requirements.txt to remove gunicorn dependency. see https://github.com/dbuscombe-usgs/dash_doodler/issues/1.
* updates to docs, README, videos

10/29/20
* switched to a Flask backend server
* added upload button/drap-and-drop
* those images now download into the assets folder and get listed in the dropdown menu
* figured out how to serve using gunicorn (`gunicorn -w 4 -b 127.0.0.1:8050 app:server`)
* results are now handled per session, with the results written to a folder with the timestamp of the session start
* organized files so top level directory has the main files, and then subfunctions are in `src`. conda and pip files are in `install`
* no more banner and instructions - saves space. Instructions on github.
* more economical use of space in drop down list - can load and display more files
* when an image is labeled, it disappears from the list when new images are uploaded
* new videos

11/04/20
* fixed bug in how annotations were being generated and written to png file
* new version uses a cumulatively trained random forest. First image annotated, builds RF, makes prediction. Then subsequent images build upon the last RF. Uses scikit-learn RandomForestClassifier's `warm_start` parameter and saving model to pickle file
* all the CRF functions and controls are removed in `appRF.py`. Just RF is implemented. Faster, also perhaps better (more testing)
* fixed bug that was making it redo segmentations automatically on file change and other callbacks
* time strings now ISO conformable (thanks Dan Nowacki)
* image printing now in the main app function rather than subfunction, aids incorporation into more sophisticated callback workflows

11/10/20
* added sliders for CRF and RF downsample factors
* fixed bug that causes error when no files present upon launch
* fixed bug that caused incorrect colormap on color label outputs where nodata also present (orthomosaics)
* automatically selects colormap based on number of classes (G10 for up to 10, Light24 for up to 24)
* tested on large set of orthomosaics and other types of imagery
* 6-stack no longer used in CRF (too slow). Back to RGB. Later will add 6-stack as an option, or individual bands as options (like for RF)
* resize now uses nearest nearest (order 0 polynomial) interpolation rather than linear. Makes more sense for discrete values
* callback context now passed to segmentation function
* `app.py` is now `appyCRF.py` and now uses CRF with fixed RF inputs, and different defaults for MU and THETA
* median filter size adjustments no longer force redoing of segmentation. Instead, it disables the segmentation so after you have set the new median filter kernel size, recheck the compute/show segmentation box

02/09/21
* Uses two tabs: tab 1 is for image annotations and controls, and tab 2 is for file selection and instructions
* RF model updating properly implemented. RF model file name contains classes
* CRF version fully exposes all parameters to user
* optionally, users can enter their unique ID for saving results
* Log file created for a session
* Better instructions
* RF-only version no longer available. Only CRF. App now called `doodler.py`
* Images that are done are copied to the 'labeled' folder. Every 2 seconds the program checks the assets and labeled folders and only lists the difference of those two sets.
* Image names are copied below into a box, for copy/pasting (note taking). The names of images done are copied to a text file, for later use

02/12/21
* Re-implemented RF updating by writing out data to file, then refitting to all data
* some argument passing changes
* RF now uses intensity, edges and texture by default (no choice)
* updated `predict_flder.py` script - first usable version

02/12/21
* If only one class example provided, whole scene assumed to be that class
* Checks for old classifiers and data files and renames (restarts each new session)
* By default now only uses 3 trees per image to update the RF model
* Each setting default now displayed on control panel
* class_weight="balanced_subsample", min_samples_split=3

03/17/21. *Release* version 1.1.0
* max samples now 1e5, subsampled thereafter from all values (new and concatenated from file)
* crf_refine now loops through 5 different 'rolled' versions of the label/image combo and and an average is taken. Rolling (wot I made up) is shifting an image and unary potentials on the x axis and recomputing the label raster in the shifted position, then unrolling back and averaging down the stack of unrolled label rasters
* min_samples_split=5 in RF
* in `predict_folder.py`, now loops through 5 different 'rolled' versions of the label/image combo and a weighted average is taken
* added a 'sample' set to test `predict_folder.py`, and provide model
* provided a 'clean' yml file (no version numbers) and added tqdm as a dependency
* program now reads in default values from `src\defaults.py` or `my_defaults.py`(see below)
* program uses SIGMA_MAX and SIGMA_MIN from  `src\defaults.py` or `my_defaults.py`(see below) rather than hard-coded in
* created `refine_labels.py` (not yet documented)
* program now writes and reads `my_defaults.py` that keeps track of your settings preferences
* IP address (http://127.0.0.1:8050/) now displayed in terminal window
* added example workflow for sample dataset

03/20/21. version 1.1.1
* support for 1-band (greyscale) images (and tested on sidescan imagery)
* adds rudimentary metric for RF and CRF to use space, the i.e. the pixel locations in x and y. seems to improve predictions in sidescan and water masking imagery
* added some explanatory text to README on how Doodler works
* increased max samples to 500,000
* DEFAULT_CRF_DOWNSAMPLE = 3 (up from 2) to accomodate larger feature stack (two more, position in x, and position in y)
* added more logging info in RF/VRF models
* added timer to show how long each inference takes

05/09/21. version 1.2.1 (MAJOR UPGRADE)
GUI:
* `Model independence factor ` and `blur factor` now used for mu and theta respectively. More approachable, easier to explain
* reordered crf controls so theta/mu, then downsample and probability of doodle (order of likelihood to tweak)
* no longer median filter controls
* made 'show/compute seg' button blue :)
* no longer the sigma for 'sigma range'

Modeling:
* per image standardization and rescaling [0,1]
* no antialiasing when resizing CRF label, and replacement of values not present in original
* decreased max samples to 200,000
* remove small holes and islands in the one-hote encoded CRF mask
* median filtering now removed. not needed, creates problems, extra buttons/complexity. Instead ...
* implements 'one-hot encoded mask spatial filtering'
* implements inpainting on regions spatially filtered
* pen width is used as-is, no longer exponentially scaled
* SIGMA_MAX=16; SIGMA_MIN=1. Hardcoded. Easier to manage number of features, which now have to be 75. Also, they make very little difference

I/O:
* greyscale and annotations no longer saved to png file, instead to numpy area (npz compressed), which encodes
  * 'image'' = image
  * 'label' = one-hot-encoded label array
  * 'color_doodles' = color 3D or color doodles
  * 'doodles' = 2D or greyscale doodles
  * the npz file is overwritten, but old arrays are kept, prefixed with '0', and prepended with another '0', such that the more '0's the newer, but the above names without '0's are always the newest. Color images are still produced with time tags.
* DEFAULT_CRF_DOWNSAMPLE = 4 by default
* accepts jpg, JPG, and jpeg
* in implementation using `predict_folder.py`, user decides between two modes, saving either default basic outputs (final output label) or the full stack out outputs for debugging or optimizing
* in `predict_folder`, extracted features are memory mapped to save RAM

Other:
* RF feature extraction now in parallel
* CRF 'test time augmentation' now in parallel
* `utils/plot_label_generation.py` is a new script that plots all the minutae of the steps involved in label generation, making plots and large npz files containing lots of variables I will explain later. By default each image is modeled with its own random forest. Uncomment "#do_sim = True" to run in 'chain simulation mode', where the model is updated in a chain, simulating what Doodler does.
* `utils/convert_annotations2npz.py` is a new script that will convert annotation label images and associated images (created and used respectively by/during a previous incarnation of Doodler)
* `utils/gen_npz_4_zoo.py` is a new script that will strip just the image and one-hot encoded label stack image for model training with Zoo

Website:
https://dbuscombe-usgs.github.io/dash_doodler/


06/01/21. version 1.2.2
* versions with S3 integration: `usgs_only_server.py` and `doodler_server.py`, in which:
  * remove all code to do with timing and file lookup in assets and labeled
  * fsspec read file
  * fsspec write results
  * 'one chance" doodling. Next image retrieved automatically from s3 when 'segment image' unchecked
* the minimal version has one chance doodling, no controls except pen width. Next image retrieved automatically from s3 when 'segment image' unchecked
* worked out more details for serving using gunicorn/nginx/systemctl services

06/06/21. `docker-dev` branch
* make better requirements.txt using `pipreqs` (https://pypi.org/project/pipreqs/)
* Dockerfile

06/09/21. v 1.2.3
* fixed bug in CRF making all labels a minimum of 1

06/21/21. v 1.2.4
* Doodler no longer learns as it goes by default. Large trials suggested that strategy was inferior to an extremely task-specific approach.
* Doodler now uses a MLP classifier using features - applies gaussian blur to image and x,y location for feat extraction as inputs to MLP with 2 hidden layers each with 100 neurons, relu activation, alpha=2 regularization
* applies standard scaler as pre-filter
* no predict in folder script
* partially fixed bug with file select (interval just 200 milliseconds)
* cleaned up and further tested all utils scripts

07/30/21. v 1.2.5
* code tidied up and commented, added docstrings
* removed last traces of RF code implementation
* added logging details, including RAM utilization throughout
* CRF and feature extraction only happen in parallel now when RAM < 10GB and usage is <50%, i.e. there is 5GB of available RAM for parallel processing
* %d-%m-%Y-%H-%M-%S changed to sortable %Y-%m-%d-%H-%M-%S format in log file
* reorganized code into modular components
* moved assets to downloadable zipped file than downloads and unpacks automatically
* removed samples
* moved gifs to a release, so could be linked in this README (used a lot of space, large download)
* two new dependencies, Flasking-Cahing, and psutil
* flask-caching is for caching - clear cache for the program independently of the browser by deleting files in the app_files/cache_directory
* now has a `.gitignore` file to ignore cached files

This is a major update and may require a new clone.

Changes include:
* revised file structure, to be modular/easier to read and parse
  * new directory `app_files` contains `cache-directory`, `logs`, and `src`
  * `logs` contains program logs - now reports RAM usage throughout program for troubleshooting. Requires new dependency `psutil`
  * `src` contains the main program code, including all of the segmentation codes (in `image_segmentation.py` and `annotations_to_segmentations.py`) and most of the app codes (`app_funcs.py` and `plot_utils.py`)
  *  `cache-directory` - clear cache for the program independently of the browser by deleting files here. Requires new dependency `flask-caching`
  * `assets` comes with no imagery, but sample imagery is, by default, downloaded automatically into that folder. You can disable the automatic downloading of the imagery by editing `environment/settings.py` where indicated
  * `install` folder is now called `environment`
  * removed the large gifs from the `assets/logos` folder (they now render in html from a github release)
  * you still run `doodler.py`, but most of the app is now in `app.py`, and a lot more of the functions are in the `app_files/src/` functions. This allows for better readability and modularity, therefore portability into other frameworks, and more effective troubleshooting
* example results files are now downloadable [here](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/example-results/results2021-06-21-11-05.zip) rather than shipped by default with the program. Run `python download_sample_results.py` from the `results` folder
* overall the download is much, much smaller, and will break less with `git pull` because the `.gitignore` contains the common gotchas
* removed superfluous buttons in the modebar
* code commented better and better docstrings and README details
* added example Dockerfile
* ports, IPS and other deployment variables can be changed or added to `environment\settings.py` that gets imported at startup
* added [Developer's Notes](#developers) to this README, with more details about program setup and function


09/30/21. v 1.2.6
(some of these in response to USGS colleague code review by Frank Engel, USGS)
* minor fixes to all utils scripts
* added 'number of scales' functionality to help with low-resource machines and give greater flexibility to the user
* the program will now automatically download the sample images by default, but only if there are no (jpeg) images in the assets folder
