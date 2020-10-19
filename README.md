
> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

> This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation and is based on code previously contained in the "doodle_labeller" [repository](https://github.com/dbuscombe-usgs/doodle_labeller) which implemenets a similar algorithm in OpenCV

> The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)


Note this tool is still under development. Please use the issues tab to report bugs and suggest improvements. Please get in touch if you're interested in helping improve this tool!

The video shows a basic usage of doodler. 1) Annotate the scene with a few examples of each class (colorful buttons).  2) Check 'compute and show segmentation' and wait for the result. The label image is written to the 'results' folder. you can also download a version of it from your browser for quick viewing

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme1.gif)


<!-- Please go to the [project website](https://dbuscombe-usgs.github.io/dash_doodler/) for more details and documentation -->

## Rationale
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

What is generally required in the above case is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image. The number of pixels annotated by the human annotator is typically a small fraction of the total pixels in the image.  

`Doodler` is a tool for "exemplative", not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more. Your annotations are first used to train and apply a random forest on the entire image, then a CRF is used to refine labels further based on the underlying image.

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, the user places imagery in the `assets` folder and creates a `classes.txt` file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. There is no limit to the maximum number of classes, except screen real estate! Label images are written to the `results` folder.


## Installation

Install the requirements

```bash
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r requirements.txt
```


## Use
Move your images into the `assets` folder. For the moment, they must be jpegs

Run the app. An IP address where you can view the app in your browser will be
displayed in the terminal.

```bash
python app.py
```

Results (label images and annotation images) are saved to the `results/` folder. You should move your images (inputs and outputs) to another place, to keep things manageable. Later versions of this tool might provide a better file management system.


## Videos


#### Video 1
Make a mistake, highlight and erase, then start labeling again. Check 'compute segmentation' and wait for the segmentation output. It needs improvement; uncheck 'compute segmentation', make more annotations and adjustments to parameters, then recompute segmentation.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme2.gif)

#### Video 2
Label an image, view the result. Make adjustments to parameters a few times, before sownloading the segmentation and viewing it in an image viewer.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme3.gif)

#### Video 3
Use a different image and class set. Annotate an image with the new 6-class set. View the resulting label image in an image editor (GIMP), and finally touch up a label image manually in the image analysis software.

![Doodler](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler_readme4.gif)


## Acknowledgements

Based on [this plotly example](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation) and the previous openCV based implementation [doodle_labeller](https://github.com/dbuscombe-usgs/doodle_labeller)
