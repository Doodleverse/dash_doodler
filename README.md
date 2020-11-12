
> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

> This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation and is based on code previously contained in the "doodle_labeller" [repository](https://github.com/dbuscombe-usgs/doodle_labeller) which implemenets a similar algorithm in OpenCV

> The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)

The video shows a basic usage of doodler. 1) Annotate the scene with a few examples of each class (colorful buttons).  2) Check `compute and show segmentation` and wait for the result. The label image is written to the `results` folder, and you can also download a version of it from your browser for quick viewing.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_nov10-a.gif)


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

`Doodler` is a tool for sparse, not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more. Your annotations are first used to train and apply a random forest on the entire image. In another version (the recommended one), then a CRF is used to refine labels further based on the underlying image.

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, a `classes.txt` file tells the program what classes will be labeled. The images that you upload will go into the `assets/` folder. The labels you create are written to the `results` folder.


## <a name="install"></a>Installation

Clone/download this repository (the `--depth 1` will clone only the latest copy of the relevant files to save space and time) 

```
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git
```

cd into the new repo and install the requirements

```bash
conda env create --file install/dashdoodler.yml
conda activate dashdoodler
```


*If* the above doesn't work, try this:

```bash
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r install/requirements.txt
```


## <a name="use"></a>Use
The general steps to using the Dash app. 

#### Adding Imagery
The two current options for loading imagery to the Dash Doodler tool:
1. Move your images into the `assets` folder - if the app has already been launched, refresh to see newly added images.
2. Launch the app and use the `Drag and drop or click to select a file to upload` option. This will also add the image to the `assets` folder. 

For the moment, they must be jpegs with the `.jpg` extension. Support for other image types forthcoming ...
Best results occur for images 3000 x 3000 pixels or less. Very large images may not load correctly and take a much longer time to compute the segmentation.

#### Classes
Edit/create the `classes.txt` file in the repository. This file tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. There is no limit to the maximum number of classes, except screen real estate! If you have more than 10 classes, the program uses `Light24` instead. This will give you up to 24 classes. Remember to keep your class names short, so the buttons all fit on the screen!

The default class colormap in the App is plotly's G10, found [here](https://plotly.com/python/discrete-color/). The hex (rgb) color sequence is:

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

(you can google search those hex codes and get a color picker view). 

Classes are discrete labels that you define to represent and reduce the dimensionality of featyres and objects in your imagery. Each class really represents a spectrum of textures, colors, and spatial extents. It can be difficult to define a good set, but the golden rule (in my humble opinion) is that each class in the set of classes should represent a spectrum of image features that collectively have much greater inter-class variability than intra-class variability.

#### App Algorithm Types
Run the app using one of the versions below. An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Tested so far with Chrome, Firefox, and Edge.

There are two versions that implement different algorithms. When in doubt, start with the Conditional Random Field (CRF) version:

```bash
python appCRF.py
```
This version currently allows for four parameters/coefficents of the segmentation to be changed by using the sliders in the app. Refreshing the app will cause the parameters to reset to their defaults, shown in bold after the range below. 

- Blurring for image feature extraction (10-220, **40**)
- Color class difference tolerance (1-255, **100**)
- Downscale factor (1-6, **2**)
- Median filter kernel radius (0-100, **3**)

The alternative is the Random Forest (RF) version

```bash
python appRF.py
```
This version currently allows for three parameters/coefficents of the segmentation to be changed by using the sliders in the app. Refreshing the app will cause the parameters to reset to their defaults, shown in bold after the range below. 

- Median filter kernel radius (0-100, **5**)
- Image Feature Extraction: 
- [x] Intensity
- [ ] Edges
- [x] Texture
- Blurring for image feature extraction (1-30, **1-16**)
- Downscale factor (2-30, **10**)

It works in a similar way, and is faster with fewer options, but is generally not as powerful. However, the algorithm may suit certain situations better than others, so you should ideally try both.

#### Making Annotations and Computing Segmentation
When first launching the app, the `Compute/Show segmentation` is de-selected. 

1. Select the image to annote/label from the `Select` dropdown. 
2. Click the class to start annotating
3. Pen width can be changed to doodle on more or less area at a time
4. Doodle! A colored line based on the class will appear. This can be selected and the nodes dragged to rearrange the doodle or deleted with the `Erase active shape` menu button
5. Use the rest of the class labels to doodle until satisfied
6. Select the `Compute/Show segmentation`
7. Wait - this could take seconds or minutes depending on the image size and processing power of your machine
8. Examine the segmentation results - if unhappy with the results there are two main options
   - add more doodles
   - change the CRF/RF parameters/coefficients
Before either, consider de-selecting `Compute/Show segmentation`, otherwise everytime you change a setting or make a single doodle the segmentation will re-compute. This can be aggravating if you want to change multiple things at once. If you wish to compare segmentation results consider using the `DOWNLOAD LABEL IMAGE`, otherwise the labeled image in the `results` folder overwrites itself. 
9. When satisfied with segmentation, select a new image to annotate. Don't forget to de-select the `Compute/Show segmentation` before starting to doodle again. 


## <a name="outputs"></a>Outputs
Results (label images and annotation images) are saved to the `results/` folder. The program creates a subfolder each time it is launched, timestamped. That folder contains your results images for a session. Each classified image will result in three files within the `/results` folder, with XXXXXXXX_ representing the image filename root:

* `XXXXXXXX_label.png`: color version of the label image
* `XXXXXXXX_label_greyscale.png`: greyscale version of the above. Note this will always appear very dark because the full range of an 8-bit image is 0 to 255. Your classes will be represented as integers
* `XXXXXXXX_annotations.png`: this is mostly for debugging/analysis and may disappear in a future version. It shows your doodles.

### Videos

Here's a video of the Random Forest version being used.

![RF version](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_nov10-b.gif)

More demonstration videos:

![Elwha example](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_nov10-c.gif)

Recheck compute segmentation when median filter changed:

![Beach example](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_nov10-d.gif)

<!-- ![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_video3.gif)

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_video4.gif)

Longer example, consisting of 5 images labeled, then add two more (see the list refresh) and label 2 more:

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_longvideo.gif) -->
 -->


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


## <a name="progress"></a>Progress report

10/20/20:
* display numbers for every parameter
* fixed label creation and display in situations where there is a null image value (0), and where there are not as many classes in the scene as in the collection

10/22/20
* modified layout so image window is larger, and button bank is narrower. Hopefully easier to label, less zooming, etc. see https://github.com/dbuscombe-usgs/dash_doodler/issues/4. Thanks Dan Nowacki
* added yml installation file, modified requirements.txt to remove gunicorn dependency. see https://github.com/dbuscombe-usgs/dash_doodler/issues/1. Thanks Dan Nowacki, Chris Sherwood, Rich Signell
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

## <a name="roadmap"></a>Roadmap

* Maybe a button to reset the coefficients to the defaults? [here](https://github.com/dbuscombe-usgs/dash_doodler/issues/2)

* Delay running the model until all of the coefficients are adjusted...right now it jumps right into the calcs as soon a slider is moved, but maybe you want to adjust two sliders first. Maybe change the compute segmentation to a button that changes color if the model is out of date wrt to the current settings. [here](https://github.com/dbuscombe-usgs/dash_doodler/issues/2)

* userID written to results files. What would this require? Text box? login account?

* pymongo (mongoDB) database backend - thanks Evan and Shah @UNCG-DAISY!

Use the issues tab to suggest new features!
