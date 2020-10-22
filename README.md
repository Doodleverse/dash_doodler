
> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

> This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation and is based on code previously contained in the "doodle_labeller" [repository](https://github.com/dbuscombe-usgs/doodle_labeller) which implemenets a similar algorithm in OpenCV

> The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)


Note this tool is still under development. Please use the issues tab to report bugs and suggest improvements. Please get in touch if you're interested in helping improve this tool!

The video shows a basic usage of doodler. 1) Annotate the scene with a few examples of each class (colorful buttons).  2) Check 'compute and show segmentation' and wait for the result. The label image is written to the 'results' folder. you can also download a version of it from your browser for quick viewing

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme1.gif)


## Contents
* [Rationale](#rationale)
* [Installation](#install)
* [Use](#use)
* [Outputs](#outputs)
* [Acknowledgments](#ack)
* [Contribute](#contribute)
* [Progress](#progress)
* [Roadmap](#roadmap)

<!-- Please go to the [project website](https://dbuscombe-usgs.github.io/dash_doodler/) for more details and documentation -->

## <a name="rationale"></a>Rationale
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

What is generally required in the above case is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image. The number of pixels annotated by the human annotator is typically a small fraction of the total pixels in the image.  

`Doodler` is a tool for "exemplative", not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more. Your annotations are first used to train and apply a random forest on the entire image, then a CRF is used to refine labels further based on the underlying image.

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, the user places imagery in the `assets` folder and creates a `classes.txt` file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. There is no limit to the maximum number of classes, except screen real estate! Label images are written to the `results` folder.


## <a name="install"></a>Installation

Clone/download this repository

```
git clone https://github.com/dbuscombe-usgs/dash_doodler.git
```

Install the requirements

```bash
conda env create --file dashdoodler.yml
conda activate dashdoodler
```


*If* the above doesn't work, try this:

```bash
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r requirements.txt
```


## <a name="use"></a>Use
Move your images into the `assets` folder. For the moment, they must be jpegs with the `.jpg` extension. Support for other image types forthcoming ...

Run the app. An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Tested so far with Chrome, Firefox, and Edge.

```bash
python app.py
```

Results (label images and annotation images) are saved to the `results/` folder. You should move your images (inputs and outputs) to another place, to keep things manageable. Later versions of this tool might provide a better file management system.

The default colormap is plotly's G10, found [here](https://plotly.com/python/discrete-color/). The hex (rgb) color sequence is:

* #3366CC (51, 102, 204)
* #DC3912 (220, 57, 18)
* #FF9900 (255, 153, 0)
* #109618 (16, 150, 24)
* #990099 (153, 0, 153)
* #0099C6 (0, 153, 198)
* #DD4477 (221, 68, 119)
* #66AA00 (102, 170, 0)
* #B82E2E (184, 46, 46)
* #316395 (49, 99, 149)

(you can google search those hex codes and get a color picker view). If you need more than 10 colors, replace `G10` with `Light24`. This will give you up to 24 classes. Remember to keep your class names short, so the buttons all fit on the screen!

### Videos


#### Video 1
Make a mistake, highlight and erase, then start labeling again. Check 'compute segmentation' and wait for the segmentation output. It needs improvement; uncheck 'compute segmentation', make more annotations and adjustments to parameters, then recompute segmentation.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme2.gif)

#### Video 2
Label an image, view the result. Make adjustments to parameters a few times, before sownloading the segmentation and viewing it in an image viewer.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme3.gif)

#### Video 3
Use a different image and class set. Annotate an image with the new 6-class set. View the resulting label image in an image editor (GIMP), and finally touch up a label image manually in the image analysis software.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme4.gif)

#### More
![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_oct20a.gif)

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_oct20b.gif)

## <a name="outputs"></a>Outputs
Each classified image will result in three files within the `/results` folder, with XXXXXXXX_ representing the image filename root:

* `XXXXXXXX_label.png`: color version of the label image
* `XXXXXXXX_label_greyscale.png`: greyscale version of the above. Note this will always appear very dark because the full range of an 8-bit image is 0 to 255. Your classes will be represented as integers
* `XXXXXXXX_annotations.png`: this is mostly for debugging/analysis and may disappear in a future version. It shows your doodles.

## <a name="ack"></a>Acknowledgements

Based on [this plotly example](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation) and the previous openCV based implementation [doodle_labeller](https://github.com/dbuscombe-usgs/doodle_labeller)

## <a name="contribute"></a>Contributing
Contributions are welcome, and they are greatly appreciated! Credit will always be given.

#### Report Bugs

Report bugs at https://github.com/dbuscombe-usgs/dash_doodler/issues.

Please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### Write Documentation

We could always use more documentation, whether as part of the docs, in docstrings, or using this software in blog posts, articles, etc.

#### Get Started!

Ready to contribute? Here's how to set up for local development.

* Fork the dash_doodler repo on GitHub.

* Clone your fork locally:

`$ git clone git@github.com:your_name_here/dash_doodler.git`

Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

`$ cd dash_doodler/`
`$ conda env create --file dashdoodler.yml`
`$ conda activate dashdoodler`

Create a branch for local development:

`$ git checkout -b name-of-your-bugfix-or-feature`

Now you can make your changes locally.

Commit your changes and push your branch to GitHub:

`$ git add .`
`$ git commit -m "Your detailed description of your changes."`
`$ git push origin name-of-your-bugfix-or-feature`

Submit a pull request through the GitHub website.


## <a name="progress"></a>Progress report

10/20/20:
* display numbers for every parameter
* fixed label creation and display in situations where there is a null image value (0), and where there are not as many classes in the scene as in the collection

10/22/20
* modified layout so image window is larger, and button bank is narrower. Hopefully easier to label, less zooming, etc. see https://github.com/dbuscombe-usgs/dash_doodler/issues/4. Thanks Dan Nowacki
* added yml installation file, modified requirements.txt to remove gunicorn dependency. see . Thanks Dan Nowacki, Chris Sherwood,
* updates to docs, README, videos

## <a name="roadmap"></a>Roadmap


* Maybe a button to reset the coefficients to the defaults?

* Delay running the model until all of the coefficients are adjusted...right now it jumps right into the calcs as soon a slider is moved, but maybe you want to adjust two sliders first. Maybe change the compute segmentation to a button that changes color if the model is out of date wrt to the current settings.

* Publish the color order so people can assign colors by changing the order of the labels.

Use the issues tab to suggest new features!
