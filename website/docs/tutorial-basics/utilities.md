---
sidebar_position: 3
---

# Utility scripts

Doodler is compatible with my other segmentation program, [Zoo](https://github.com/dbuscombe-usgs/segmentation_zoo) in a couple of different ways:

* You could run the function `gen_npz_4_zoo.py` to create npz files that contain only image and label pairs. This is the same output as you would get from running the Zoo program `make_datasets.py'

* You could alternatively run the function `gen_images_and_labels_4_zoo.py` that would generate jpeg greyscale image files and image jpegs for use with the Zoo program `make_datasets.py'.

The first scenario might be most common because it requires one less step, however the second scenario might be useful for using the labels with another software package, or for further post-processing of the labels

There are two additional scripts in the `utils` folder:

* `viz_npz.py` creates transparent overlay plots of images and labels, and has three modes with the following syntax `viz_npz.py [-t npz type {0}/1/2]` where optional `-t` controls what type of npz file: native from doodler (option 0, default), a `labelgen` file from `plot_label_generation.npz`, a npz file used as input for Zoo

* `plot_label_generation.py` that generates a detailed sequence of plots for every input npz file from doodler, including plots of the doodles themselves, overlays, and internal model outputs.
