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
Check out the [Doodler website](https://doodleverse.github.io/dash_doodler/)

### Paper
[![Earth ArXiv Preprint
DOI](https://img.shields.io/badge/%F0%9F%8C%8D%F0%9F%8C%8F%F0%9F%8C%8E%20EarthArXiv-doi.org%2F10.31223%2FX59K83-%23FF7F2A)](https://doi.org/10.31223/X59K83)

Buscombe, D., Goldstein, E.B., Sherwood, C.R., Bodine, C., Brown, J.A., Favela, J., Fitzpatrick, S., Kranenburg, C.J., Over, J.R., Ritchie, A.C. and Warrick, J.A., 2021. Human‐in‐the‐Loop Segmentation of Earth Surface Imagery. Earth and Space Science, p.e2021EA002085[https://doi.org/10.1029/2021EA002085](https://doi.org/10.1029/2021EA002085)

### Code that made the paper
[![DOI](https://zenodo.org/badge/304798940.svg)](https://zenodo.org/badge/latestdoi/304798940)

### Data that made the paper
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.2fqz612ps)

### Overview
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](https://www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

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

> Check out the installation guide on the [Doodler website](https://doodleverse.github.io/dash_doodler/docs/tutorial-basics/deploy-local)


We advise creating a new conda environment to run the program.

1. Clone the repo:

```
git clone --depth 1 https://github.com/Doodleverse/dash_doodler.git
```

(`--depth 1` means "give me only the present code, not the whole history of git commits" - this saves disk space, and time)

2. Create a conda environment called `dashdoodler`

```
conda env create --file environment/dashdoodler-clean.yml
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
conda install -c conda-forge pydensecrf cairo cairosvg
conda install -c conda-forge scikit-learn scikit-image psutil dash flask-caching requests
pip install -r environment/requirements.txt
```

and good luck to you!

## 🚀 Usage

> Check out the user guide on the [Doodler website](https://doodleverse.github.io/dash_doodler/docs/tutorial-basics/what-to-do)

Move your images into the `assets` folder. For the moment, they must be jpegs with the `.jpg` (or `JPG` or `jpeg`) extension. Support for other image types forthcoming ...

Run the app. An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Tested so far with Chrome, Firefox, and Edge.

```bash
python doodler.py
```

Open a browser and go to 127.0.0.1:8050. You may have to hit the refresh button. If, after some time doodling things seem odd or buggy, sometimes a browser refresh will fix those glitches.

### Example screenshots of use with example dataset

<!-- (note: these are screengrabs of an older version of the program, so the buttons and their names are now slightly different) -->

#### `doodler.py`

Videos showing Doodler in action:

![Doodler example 2](https://github.com/Doodleverse/dash_doodler/releases/download/gifs/doodler_demo_simple_watermask_v2.gif)

![Doodler example 3](https://github.com/Doodleverse/dash_doodler/releases/download/gifs/doodler_demo_sediment_v2.gif)

<!-- ![Example 1](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py.png)
![Example 2](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py1.png)
![Example 3](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py2.png)
![Example 4](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py3.png)
![Example 5](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py4.png)
![Example 6](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_py5.png) -->

<!-- ### Videos
More demonstration videos (older version of the program): -->

<!-- ![Doodler example 2](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/quick-saturban-x2c.gif)

![Doodler example 3](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short.gif)

![Elwha example](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-elwha.gif)

![Coast Train example](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-coast.gif)

![Coast Train example 2](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-coast2.gif) -->


### <a name="coasttrain"></a>Unpacking Coast Train Data

To use the labels in their native class sets (that vary per image), use the `gen_images_and_labels.py` script as described below. To use the labels in remapped classes (standardized across image sets), use the `gen_remapped_images_and_labels.py` script described below.


### <a name="utilities"></a>Utility scripts

Doodler is compatible with the partner segmentation program, [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) in a couple of different ways:

1. You could run the function `gen_npz_4gym.py` to create npz files that contain only image and label pairs. This is the same output as you would get from running the Gym program `make_nd_datasets.py'

2. You could alternatively run the function `gen_images_and_labels.py` that would generate jpeg greyscale image files and label image jpegs for use with the Gym program `make_nd_datasets.py'.

3. Finally, you could run the function `gen_remapped_images_and_labels.py` that would generate jpeg greyscale image files and remapped label image jpegs for use with the Gym program `make_nd_datasets.py'. Labels are remapped based on a dictionary of class aliases and a list of classes present, using a special config file. To remap Coast Train data, use the config files provided [here](https://github.com/dbuscombe-usgs/CoastTrainMetaPlots/tree/main/remap_config_files) 

The first scenario might be most common because it requires one less step, however the second scenario might be useful for using the labels with another software package, or for further post-processing of the labels

There are additional scripts in the `utils` folder:

1. `viz_npz.py` creates transparent overlay plots of images and labels, and has three modes with the following syntax `viz_npz.py [-t npz type {0}/1/2]` where optional `-t` controls what type of npz file: native from doodler (option 0, default), a `labelgen` file from `plot_label_generation.py`, a npz file used as input for Gym

2. `plot_label_generation.py` that generates a detailed sequence of plots for every input npz file from doodler, including plots of the doodles themselves, overlays, and internal model outputs.

3. `gen_overlays_from_images_and_labels.py` that generates color overlay figures from folders of images and greyscale labels

4. `gen_remapped_images_and_labels.py` that generates remapped label images from one class set to another


## 💭 Feedback and Contributing

Please read our [code of conduct](https://github.com/Doodleverse/dash_doodler/blob/main/CODE_OF_CONDUCT.md)

Please contribute to the [Discussions tab](https://github.com/Doodleverse/dash_doodler/discussions) - we welcome your ideas and feedback.

We also invite all to open issues for bugs/feature requests using the [Issues tab](https://github.com/Doodleverse/dash_doodler/issues)


### <a name="contribute"></a>Contributing
Contributions are welcome, and they are greatly appreciated! Credit will always be given.

#### Report Bugs

Report bugs at https://github.com/Doodleverse/dash_doodler/issues.

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

> See the [how to contribute](https://Doodleverse.github.io/dash_doodler/docs/tutorial-extras/how-to-contribute) section of the Doodler website

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

