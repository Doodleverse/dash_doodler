---
sidebar_position: 4
---

# [ADVANCED] Next steps .... training a Deep Learning model for image segmentation

## Utility scripts

Doodler is compatible with the segmentation program, [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) in a couple of different ways:

### Scenario 1 (most common)

You wish to generate labels and images for subsequent use in Gym, specifically to make a dataset to train or test an image segmentation model

1. run `gen_images_and_labels.py`

This will generate 4 folders, 1 for each of the output types

> `images`: The images you doodled

![](/img/utils/rgb.jpg)

> `labels`: The color and greyscale label images

![](/img/utils/rgblabel.png)

![](/img/utils/label.jpg)

> `doodles`: The image with the color doodles as a semi-transparent overlay

![](/img/utils/doodled.png)

> `overlays`: The image with the color label as a semi-transparent overlay

![](/img/utils/overlay.png)


### Scenario 2

You wish to use the labels and images in Zoo, but wish to bypass running the Zoo program `make_datasets.py', i.e. you'd like to pass the original labels and images to the model training rather than create augmented outputs of a certain size

run `gen_npz_4gym.py`  to create npz files that contain only image and label pairs.

