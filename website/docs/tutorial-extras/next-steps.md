---
sidebar_position: 4
---

# [ADVANCED] Next steps .... training a Deep Learning model for image segmentation

(page under construction)

### Utility scripts

Doodler is compatible with my other segmentation program, [Zoo](https://github.com/dbuscombe-usgs/segmentation_zoo) in a couple of different ways:

* You could run the function `gen_npz_4_zoo.py` to create npz files that contain only image and label pairs. This is the same output as you would get from running the Zoo program `make_datasets.py'

* You could alternatively run the function `gen_images_and_labels_4_zoo.py` that would generate jpeg greyscale image files and image jpegs for use with the Zoo program `make_datasets.py'.

The first scenario might be most common because it requires one less step, however the second scenario might be useful for using the labels with another software package, or for further post-processing of the labels

There are two additional scripts in the `utils` folder:

* `viz_npz.py` creates transparent overlay plots of images and labels, and has three modes with the following syntax `viz_npz.py [-t npz type {0}/1/2]` where optional `-t` controls what type of npz file: native from doodler (option 0, default), a `labelgen` file from `plot_label_generation.npz`, a npz file used as input for Zoo

* `plot_label_generation.py` that generates a detailed sequence of plots for every input npz file from doodler, including plots of the doodles themselves, overlays, and internal model outputs.


### Deep Learning for Image Segmentation

There are two main drawbacks with neural networks consisting of only dense layers for classifying images:

* Too many parameters means even simple networks on small and low-dimensional imagery take a long time to train

* There is no concept of spatial distance, so the relative proximity of image features or that with respect to the classes, is not considered

For both the reasons above, Convolutional Neural Networks or CNNs have become popular for working with imagery. CNNs train in a similar way to "fully connected" ANNs that use Dense layers throughout.

CNNs can handle multidimensional data as inputs, meaning we can use 3- or more band imagery, which is typical in the geosciences. Also, each neuron isn't connected to all others, only those within a region called the receptive field of the neuron, dictated by the size of the convolution filter used within the layer to extract features from the image. By linking only subsets of neurons, there are far fewer parameters for a given layer

So, CNNS take advantage of both multidimensionality and local spatial connectivity

CNNs tend to take relatively large imagery and extract smaller and smaller feature maps. This is achieved using pooling layers, which find and compress the features in the feature maps outputted by the CNN layers, so images get smaller and features are more pronounced


## U-Net

Introduced in 2015, the U-Net model is relatively old in the deep learning field (!) but still popular and is also commonly seen embedded in more complex deep learning models

The U-Net model is a simple fully convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. Mainly, it consists of two parts.

* Encoder: we apply a series of conv layers and downsampling layers (max-pooling) layers to reduce the spatial size
* Decoder: we apply a series of upsampling layers to reconstruct the spatial size of the input.
The two parts are connected using a concatenation layers among different levels. This allows learning different features at different levels. At the end we have a simple conv 1x1 layer to reduce the number of channels to 1.

U-Net is symmetrical (hence the "U" in the name) and uses concatenation instead of addition to merge feature maps

The encoder (left hand side of the U) downsamples the N x N x 3 image progressively using six banks of convolutional filters, each using filters double in size to the previous, thereby progressively downsampling the inputs as features are extracted through max pooling

A 'bottleneck' is just machine learning jargon for a very low-dimensional feature representation of a high dimensional input. Or, a relatively small vector of numbers that distill the essential information about a large image An input of N x N x 3 (>>100,000 numbers) has been distilled to a 'bottleneck' of 16 x 16 x M (<<100,000 numbers)

The decoder (the right hand side of the U) upsamples the bottleneck into a N x N x 1 label image progressively using six banks of convolutional filters, each using filters half in size to the previous, thereby progressively upsampling the inputs as features are extracted through transpose convolutions and concatenation. A transposed convolution is a relatively new type of deep learning model layer that convolves a dilated version of the input tensor, in order to upscale the output. The dilation operation consists of interleaving zeroed rows and columns between each pair of adjacent rows and columns in the input tensor. The dilation rate is the stride length

Finally, make the classification layer using one final convolutional layers that essentially just maps (by squishing over 16 layers) the output of the previous layer to a single 2D output (with values ranging from 0 to 1) based on a sigmoid activation function
