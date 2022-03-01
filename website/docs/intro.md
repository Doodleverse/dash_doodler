---
sidebar_position: 1
---

# What is Doodler for?


Doodler is a web application for image segmentation, which is the process of turning this (for example, or almost any type of image):

![](/img/ex_image.jpg)

into this:

![](/img/ex_label.png)

which is a corresponding label image where each pixel is coded a single class, shown as different colors (blue=sky, red=water, yellow=surf, green=sand, purple=vegetation) ...

Doodler does this via this:

![](/img/ex_annos.png)

-- the "doodles" you made to give the program examples of each class. Doodler makes a 'model' of the image features that corresponds to each class, then completes the scene by assigning a modeled class to each pixel.

Doodler is designed to do two things:

1. It allows you to carry out image segmentation, providing a quicker (and possibly less exact) alternative to polygonal (i.e. object-based) labeling programs
> The primary purpose of Doodler is to semi-automate the process of generating the label imagery required to train start-of-the-art deep learning models for image segmentation can require hundreds to tens of thousands of example label images.

2. It uses the information you provide (i.e. your 'doodles' or sparse mouse/stylus annotations) to build a Machine Learning model for subsequent application on sample imagery
> This might suit somebody who does not want or need to train a model to achieve the same result completely automatically. You may only have a few images to label; Doodler is perfect for this use case.


For natural and other scenes, doodler can be a relatively quick (in terms of the hours you spend annotating) way to generate large numbers of label images. For high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. Doodler offers a potential alternative.


### Using Doodler

Here's a movie of Doodler in action:

![](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/short_1024px_30fps.gif)

There are two tabs, one for 'doodling' on images, and one for selecting new images from a 'to-do' list. Doodler will remove image names from the list if a label image has previously been generated. The basic workflow is:

1. Annotate the scene with a few examples of each class (colorful buttons).
2. Check 'compute and show segmentation' and wait for the result.
3. Add or remove doodles as necessary, repeating steps 1. and 2. above
4. [if still not happy with the result] Modify the parameters and repeat step 2. above

Here are more examples of Doodler (older versions) in action
![](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-elwha.gif)

![](https://github.com/dbuscombe-usgs/dash_doodler/releases/download/gifs/doodler-demo-2-9-21-short-coast2.gif)

You upload your own imagery, and define your own class set (list of class names, each of which will be assigned a different color button like in the examples above), and off you go!

### The Terms and Conditions

Before we begin, let's state some important factors around Doodler and its use

1. Doodler is research software, not commercial software. It is created by scientists with no training in software development, for the specific purposes of classifying images of natural (principally coastal) environments.
2. We hope you find it useful for your work, but we can only offer limited support. Doodler is open source code under an MIT license; you may modify this software for your own purposes, as long as you properly attribute the source (i.e. the original github repository).
3. We are preparing a manuscript describing Doodler, its uses and how it works. Once that manuscript is published, we'd appreciate you cite that paper if you find Doodler useful for your own work.
4. Doodler is made publicly available in the spirit of open source and transparent science. If you have bugs, comments, new feature requests, etc, please submit an [issue](https://github.com/dbuscombe-usgs/dash_doodler/issues) on github. Please do NOT email the authors. The primary purpose of github is not to 'deliver' you code, it is to facilitate open collaboration. Emails, which are not visible to the public, will therefore be ignored.
5. Please contribute! Submit issues, and [contribute to the code and documentation](tutorial-extras/how-to-contribute)


### Credits
Doodler is written and maintained by Dr Daniel Buscombe, Marda Science, LLC, contracted to the U.S. Geological Survey Pacific Coastal and Marine Science Center in Santa Cruz, CA. Doodler development is funded by the U.S. Geological Survey Coastal Hazards Program, and is for the primary usage of U.S. Geological Survey scientists, researchers and affiliated colleagues working on the Hurricane Florence Supplemental Project and other coastal hazards research.

Thanks to Jon Warrick, Phil Wernette, Chris Sherwood, Jenna Brown, Andy Ritchie, Jin-Si Over, Christine Kranenburg, and the rest of the Florence Supplemental team; to Evan Goldstein and colleagues at University of North Carolina Greensboro; Leslie Hsu at the USGS Community for Data Integration; and LCDR Brodie Wells, formerly of Naval Postgraduate School, Monterey. Doodler was inspired by [this](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation) plotly example and also the previous openCV based implementation called [doodle_labeller](https://github.com/dbuscombe-usgs/doodle_labeller), also written by Dr Daniel Buscombe.

See our [journal manuscript](https://doi.org/10.1029/2021EA002085) for the most comprehensive explanation for how Doodler works

Citation: Buscombe, D., Goldstein, E.B., Sherwood, C.R., Bodine, C., Brown, J.A., Favela, J., Fitzpatrick, S., Kranenburg, C.J., Over, J.R., Ritchie, A.C. and Warrick, J.A., 2021. Human‐in‐the‐Loop Segmentation of Earth Surface Imagery. Earth and Space Science, p.e2021EA002085 https://doi.org/10.1029/2021EA002085


### Sounds good! How do I install?

You have two options: use [locally on your own machine](tutorial-basics/deploy-local) or (more advanced) [deploy it on a virtual machine for others to use](tutorial-extras/deploy-server)
