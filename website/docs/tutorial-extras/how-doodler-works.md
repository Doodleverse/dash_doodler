---
sidebar_position: 1
---

# [ADVANCED] How Doodler works

(Please note that this material will be included in a forthcoming journal manuscript that describes Doodler and its uses. That manuscript is currently in preparation)

Images are labeled in sessions. During a session, a Machine Learning model is built progressively using provided labels from each image. Below, we illustrate each of the various processing steps in turn, using a single example of a 2-class labeling exercise. The two classes are 'land' (red) and 'water' (blue).

:::tip Tip

Doodler 'learns as you go'. In a Doodler session, the skill of predictions usually improves the more images you doodle in a single session. That's because it updates the model with each set if new image feature-class pairings you provide it

:::

### Overview

The figure below depicts a typical Doodler session in which images (from left to right) are labeled sequentially. The figure is read from left to right, depicting the sequential nature of image labeling, as well as from top to bottom, which depicts the sequence of processes that occur to collectively turn the image at the top, to the label image at the bottom.

![](/img/paperfig_RFchain_ann.jpg)

There is a lot going on in this figure, and it is read from top to bottom (single image) and from left to right (a sequence of images in a session), so let's break it down a little ...

### Sparse annotation or 'doodling'

It all begins with your inputs - doodles. This is what the subsequent list of operations are entirely based upon, together with some spatial logic and some assumptions regarding the types of 'features' to extract from imagery that would collectively best predict the classes.

![](/img/tutorial/D800_20160308_222129lr03-3_db_image_doodles_labelgen.png)


### Image standardization

Doodler first standardizes the images, which means every pixel value is scaled to have a mean of zero and a variance of 1, or 'unit' variance. For the model, this ensures every image has a similar data distribution. It also acts like a 'white balance' filter for the image, enhancing color and contrast.

![](/img/tutorial/D800_20160308_222129lr03-3_db_image_filt_labelgen.png)

### Feature extraction
Doodler estimates a dense (i.e. per-pixel) label image from an input image and your sparse annotations or 'doodles'. It does this by first extracting image features in a prescribed way (i.e. the image features are extracted in the same way each time) and matching those features to classes using Machine Learning.

Doodler extracts a series of 2D feature maps from the standardized input imagery. From each sample image, 75 2D image feature maps are extracted (5 feature types, across 15 unique spatial scales)

The 5 extracted feature types are (from left to right in the image below):
* Relative location: the distance in pixels of each pixel to the image origin
* Intensity: Gaussian blur over a range of scales
* Edges: Sobel filter of the Gaussian blurred images
* Primary texture: Matrix of texture values extacted over a range of scales as the 1st eigenvalue of the Hessian Matrix
* Secondary texture: Matrix of texture values extacted over a range of scales as the 2nd eigenvalue of the Hessian Matrix

In the figure below, only the 5 feature maps extracted at the smallest and largest scales are shown for brevity:

![](/img/tutorial/D800_20160308_222129lr03-3_db_image_feats_labelgen.png)


### Initial Pixel Classifier
As stated above, Doodler extracts features from imagery, and pairs those extracted features with their class distinctions provided by you in the form of 'doodles'. How that pairing occurs is achieved using Machine Learning, or 'ML' for short. Doodler uses two particular types of ML. The first is called a 'Multilayer Perceptron', or MLP for short. The seond ML model we use is called a "CRF' and we'll talk about that later.

The first image is "doodled", and the program creates a MLP model that predicts the class of each pixel according to the distribution of features extracted from the vicinity of that pixel.

Those 2D features are then flattened to a 1D array of length M, and stack them N deep, where N is the number of individual 2D feature maps, such that the resulting feature stack is size MxN. Provided label annotations are provided at a subset, i, of the M locations, M_i, so the training data is the subset {MxN}_i. That training data is further subsampled by a user-defined factor, then used to train a MLP classifier.

Below is a graphic showing, for this particular sample image we are using in this example, how the trained MLP model is making decisions based on pairs of input features. Each of the 9 subplots shown depict a 'decision surface' for the two classes (water in blue and land in red) based on a pair of features. The colored markers show actual feature values extracted from image-feature-class pairings. As you can see, the MLP model can extrapolate a decision surface beyond the extents of the data, which is useful in situations when data is encountered with relatively unusual feature values.

![](/img/tutorial/D800_20160308_222129lr03-3_db_RFdecsurf_labelgen_ann.png)

In reality, the MLP model does this on all 75 features and their respective combinations (2775 unique pairs of 75 features) simultaneously. It combines this information to predict a unique class (encoded as an integer value) for each pixel. The computations happen in 1D, i.e. on arrays of length MxN, which are then reshaped. Therefore the only spatial information used in prediction is that of the relative location feature maps.

:::tip Tip

The program adopts a strategy similar to that described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244), in that a 'global' model trained on many samples is used to provide an initial segmentation on each sample image, then that initial segmentation is refined by a CRF, which operates on a task specific level. In [Buscombe and Ritchie (2018)](references), the model was a deep neural network trained in advance on large numbers of samples and labels. Here, the model is built as we go, building progressively from user inputs. Doodler uses a MLP as the baseline global model, and the CRF implementation is the same as that desribed by Buscombe and Ritchie (2018)
:::

<!-- ### Feature importances

The relative importance of each feature for prediction is computed as the mean of accumulation of the impurity decrease within each tree. This is known as the "Gini importance" score (see [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) for more details on the implementation).

Below is a graph showing the feature importance scores for each of the 75 features for this example image. The 4 highest scores are highlighted with a dashed red line. Those 4 feature maps are shown. In this particular example, the most important features for prediction relate to image intensity extracted over different scales.

![](/img/tutorial/D800_20160308_222129lr03-3_db_RF_featimps_labelgen.png)

It is instructive to view these plots for each image prediction, in order to get a better sense of which features are considered more important. From 3 further examples shown below, it is apparent (at least in these data) that location, intensity, edges and texture are all considered important for MLP model prediction, at a range of scales

#### Example 2
In this example, location and intensity were most important, at 2 particular scales
![](/img/tutorial/featimps1.png)

#### Example 3
In this example, the features deemed most important were related to edges and texture
![](/img/tutorial/featimps2.png)

#### Example 4
In this example, location and intensity were again deemed most important, again at 2 particular scales
![](/img/tutorial/featimps3.png)

 -->

### Spatial filtering of MLP predictions
Each MLP prediction (a label matrix of integer values, each integer corresponding to a unique class). That corresponds to the left image in the figure below. You can see there is a lot of noise in the prediction; for example, and most notably, there are several small 'islands' of land in the water that are model errors. Therefore each label image is filtered using two complementary procedures that operate in the spatial domain.

![](/img/tutorial/D800_20160308_222129lr03-3_db_rf_label_filtered_labelgen.png)

The first filtering exercise (the outputs of which are labeled "b) Filtered" in the example figure above) operates on the one-hot encoded stack of labels. For each, pixel 'islands' less than a certain size are removed (filled in with the value of the surrounding area). Additionally, pixel 'holes' are also sealed (filled in with the value of the surrounding area). You can see in the example, by comparing a) and b) in the above figure, that many islands were removed in this process on this particular example.

The second filter then determines a 'null class' based on those pixels that are furthest away from similar classes. Those pixels occur at the transition areas between large contiguous regions of same-class. The process by which this is acheived is described in the figure below:

![](/img/tutorial/D800_20160308_222129lr03-3_db_rf_spatfilt_dist_labelgen_ann.png)

The intuition for 'zeroing' these pixels is to allow a further model, described below, to estimate the appropriate class values for pixels in those transition areas.

### Conditional Random Field Modeling

:::tip Tip

Already you can see how we are building a lot of robustness to natural variability:

1. images are standardized
2. Other measures are made to prevent model overfitting, such as downsampling
3. Different types of image and location features are used and extracted at a variety of scales
4. MLP outputs are filtered using relative spatial information

Next we'll go even further by making use of both global and local predictions.

The global predictions are provided by the MLP model. They are called 'global' because the model is built from all doodled images in a sequence.

The local predictions are provided by a different type of ML model, called a CRF, which is explained below.
:::

That initial MLP provides an initial estimate of the entire label estimate, which is fed (with the original image) to a secondary post-processing model based on a fully connected Conditional Random Field model, or CRF for short. The CRF model refines the label image, using the MLP model output as priors that are refined to posteriors given the specific image. As such, the MLP model is treated as a initial model and the CRF model is for 'local' (i.e. image-specific) refinement.

The CRF builds a model for the likelihood of the MLP-predicted labels based on the distributions of features it extracts from the imagery, and can reclassify pixels (it is intended to do so). Its feature extraction and decision making behavior is complex and governed by parameters. That prediction then is further refined by applying the model to numerous transformed versions of the image, making predictions, untransforming, and then averaging the stack of resulting predictions. This concept is called test-time-augmentation and is illustrated in the figure below as outputs from using 10 'test-time augmented' inputs:

![](/img/tutorial/D800_20160308_222129lr03-3_db_crf_tta_labelgen.png)


### Spatial filtering of CRF prediction

The label image that is the result of the above process is yet further filtered using the same two-part spatial procedure described above for the MLP model outputs. Usually, these procedures revert fewer pixel class values than the equivalent prior process on the MLP model outputs. The outputs are shown in 'b) and c)' in the figure below. A final additional step not used on MLP model outputs is to 'inpaint' the pixels in identified transition areas using nearest neighbor interpolation ('d)' in the figure below)

![](/img/tutorial/D800_20160308_222129lr03-3_db_crf_label_filtered_labelgen.png)

A final label image is then stored in disk:

![](/img/tutorial/D800_20160308_222129lr03-3_db_image_label_final_labelgen.png)


<!-- ## Image feature extraction and Machine Learning model -->
<!--
The figure below summarizes the feature extraction process.
![](/img/paperfig_features_ann.jpg) -->
