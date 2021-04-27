---
sidebar_position: 1
---

# What is Doodler for?

Doodler is designed to do two things:

1. It allows you to carry out image segmentation, quickly and effectively on any type of image.
> This might suit somebody who does not want or need to train a model to acheive the same result completely automatically. You may only have a few images to label; doodler is perfect for this use case.

2. It allows you to generate label data to train other types of machine learning models for image segmentation, quickly and effectively on any type of image.
> By providing enough examples of images and their corresponding pixelwise labels, models can be trained to generate the same types of segmentations on other image collections, such as future data collections in regular image-based surveys.


Training start-of-the-art deep learning models for image segmentation can require hundreds to thousands of example label images.

For natural and other scenes, doodler can be a relatively quick (in terms of the hours you spend annotating) way to generate large numbers of label images. For high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. Doodler offers a potential alternative.

### Using Doodler

This video shows a basic usage of doodler.

![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/quick-satshoreline-x2c.gif)

1. Annotate the scene with a few examples of each class (colorful buttons).
2. Check 'compute and show segmentation' and wait for the result.
3. Add or remove doodles as necessary, repeating steps 1. and 2. above
4. [if still not happy with the result] Modify the parameters and repeat step 2. above

Here are more examples of Doodler in action

![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/quick-satshore2-x2c.gif)

### Sounds good! How do I install?

You have two options: use [locally on your own machine](tutorial-basics/deploy-local) or [deploy it on a virtual machine for others to use](tutorial-extras/deploy-server)
