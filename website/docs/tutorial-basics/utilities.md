---
sidebar_position: 3
---

# Utility scripts

Doodler is compatible with the segmentation program, [Zoo](https://github.com/dbuscombe-usgs/segmentation_zoo) in a few of different ways.


### Scenario 1 (most common)

You wish to generate labels and images for subsequent use in Zoo, specifically to make a dataset to train or test an image segmentation model

> run the function `gen_images_and_labels_4_zoo.py`

This will generate 4 folders, 1 for each of the output types

1. images:
> The images you doodled

![](/img/utils/rgb.jpg)

2. labels
> The color and greyscale label images

![](/img/utils/rgblabel.png)

and greyscale

![](/img/utils/label.jpg)

3. doodles
> The image with the color doodles as a semi-transparent overlay

![](/img/utils/doodled.png)

4. overlays
> The image with the color label as a semi-transparent overlay

![](/img/utils/overlay.png)


### Scenario 2

You wish to use the labels and images in Zoo, but wish to bypass running the Zoo program `make_datasets.py', i.e. you'd like to pass the original labels and images to the model training rather than create augmented outputs of a certain size

> run the function `gen_npz_4_zoo.py` to create npz files that contain only image and label pairs.


### Scenario 3

You wish to know more details about how Doodler arrived at the solution, perhaps for troubleshooting purposes

> `plot_label_generation.py` that generates a detailed sequence of plots for every input npz file

For example, this figure shows the intermediate MLP output
![](/img/utils/MLP_output.png)

that you may wish to compare to the final CRF output (in this example they are almost identical)
![](/img/utils/CRF_output.png)

this figure shows the sequence of filters applied to the MLP output and how that gets 'inpainted' by the CRF
![](/img/utils/filter.png)

this figure shows the first and last set of five extracted features from the input image
![](/img/utils/features.png)

### Scenario 4

You simply want to see what's inside a folder of npz files:

*> `viz_npz.py` creates transparent overlay plots of images and labels, and has three modes with the following syntax `viz_npz.py [-t npz type {0}/1/2]` where optional `-t` controls what type of npz file: native from doodler (option 0, default), a `labelgen` file from `plot_label_generation.npz`, a npz file used as input for Zoo
