---
sidebar_position: 1
---

# How to Doodle

:::tip Art or Science?

While some elements of Doodler are an exact science, the act of "Doodling" itself is an art and can take some practice to get right. You should be prepared to experiment with images (and especially the size of images you use), and different class sets, to see what is most optimal for you.

:::

## A visual guide


In each of the graphics below, an image is overlain with a semi-transparent mask, color-coded by class label, as indicated by the colorbar. The image on the left is the output of the model (segmented label image) and the image on the right is the annotations (or doodles) provided by a human.

![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto1.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto2.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto3.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto4.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto5.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto6.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto7.svg)


## How to decide on 'classes'

:::danger Take care

Deciding on the optimal class set for your imagery, and the subsequent use of your label images, is tricky and may take a few attempts to get right. So start small, and be prepared to experiment.

:::

Doodler segments images into "classes" that are discrete labels that you define to represent and reduce the dimensionality of features and objects in your imagery.

Each class really represents a spectrum of textures, colors, and spatial extents. It can be difficult to define a good set, but the golden rule (in my humble opinion) is that each class in the set of classes should represent a spectrum of image features that collectively have much greater inter-class variability than intra-class variability.

That can be difficult to define in practice with generality, but here's a tip: imagine a really low- (coarse-) resolution version of your image (or make one!) and ask yourself, "are there are two sets of features that could be easily misconstrued as being in the same class, but are in fact two separate classes?" If the answer in yes, ideally you should lump those classes into a merged class. If your intended outcome dictates that's not possible to do, ask yourself if the two classes could be broken up further, say into 3 or 4 classes, to change your answer to the above question?

In general, doodler is optimized for classes that are relatively large (in terms of contiguous spatial extent) and distinct from one another.


You can have any number of classes (well, 2 or more, with the two classes case styled as something and other) but note that increasing the number of classes also increases the amount of time it takes to interact with the program, since each class must be either labelled or actively skipped.

Be prepared to experiment with classes - how well do they work on a small subset of your imagery?
Look at the unusual images in your set and and decide if there is a class there you hadn't previously thought of.
If you can't decide whether to lump or split certain classes or sets of classes, and experiment on a small test set
