---
sidebar_position: 1
---

# [ADVANCED] How Doodler works

Doodle extracts features from imagery, and pairs those extracted features with their class distinctions provided by you in the form of 'doodles'. It creates a [Random Forest](../tutorial-basics/glossary#random-forest) that predicts the class of each pixel according to the distribution of features extracted from the vicinity of that pixel.

## Overview

:::tip Tip

Doodler 'learns as you go'. In a Doodler session, the skill of predictions usually improves the more images you doodle in a single session. That's because it updates the Random Forest (RF) model with each set if new image feature - class pairings you provide it

:::

The figure below depicts a typical Doodler session in which images (from left to right) are labeled sequentially. The figure is read from left to right, depicting the sequential nature of image labeling, as well as from top to bottom, which depicts the sequence of processes that occur to collectively turn the image image at the top, to the label image at the bottom.

![](/img/paperfig_RFchain_ann.jpg)

The first image is "doodled", and the initial RF is created. That initial RF provides an initial estimate of the entire label estimate, which is fed (with the original image) to a secondary post-processing model based on a fully connected Conditional Random Field model, or CRF for short. The CRF model refines the label image, using the RF model output as priors that are refined to posteriors given the specific image. As such, the RF model is treated as a global model that receives training input from the user over multiple images, and the CRF model is for 'local' (i.e. image-specific) refinement.

![](/img/paperfig_RFchain_ann-ex1.jpg)

To recap, the program uses two Machine Learning models in concert to segment images using user-provided annotations or 'doodles'. The two models are 1) Random Forest, or RF, and 2) Fully Connected Conditional Ransom Field (CRF). As such, the program adopts a strategy similar to that described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244), in that a 'global' model trained on many samples is used to provide an initial segmentation on each sample image, then that initial segmentation is refined by a CRF, which operates on a task specific level. In [Buscombe and Ritchie (2018)](references), the model was a deep neural network trained in advance on large numbers of samples and labels. Here, the model is built as we go, building progressively from user inputs. Doodler uses a Random Forest as the baseline global model, and the CRF implementation is the same as that desribed by Buscombe and Ritchie (2018).


## Image feature extraction and Machine Learning model

Images are labeled in sessions. During a session, a RF model is initialized then built progressively using provided labels from each image. The RF model uses features extracted from the image

The figure below summarizes the feature extraction process. From each sample image, 75 image feature maps are extracted (5 feature types, across 15 unique spatial scales)
![](/img/paperfig_features_ann.jpg)

The 5 extracted feature types are ("b)" in the figure)
* Intensity: Gaussian blur over a range of scales
* Edges: Sobel filter of the Gaussian blurred images
* Primary texture: Matrix of texture values extacted over a range of scales as the 1st eigenvalue of the Hessian Matrix
* Secondary texture: Matrix of texture values extacted over a range of scales as the 2nd eigenvalue of the Hessian Matrix
* Relative location: the distance in pixels of each pixel to the image origin

Each RF prediction (a label matrix of integer values, each integer corresponding to a unique class). The CRF builds a model for the likelihood of the RF-predicted labels based on the distributions of features it extracts from the imagery, and can reclassify pixels (it is intended to do so). Its feature extraction and decision making behavior is complex and governed by parameters. The output segmentation is depicted as "c)" in the figure. That prediction then is further refined by applying the model to numerous transformed versions of the image, making predictions, untransforming, and then averaging the stack of resulting predictions. This concept is called [test-time-augmentation](../tutorial-basics/glossary#test-time-augmentation) The output segmentation is depicted as "d)" in the figure. Finally, an optional median filter is applied to the output image to spatially smooth the label.  The output segmentation is depicted as "e)" in the figure.
