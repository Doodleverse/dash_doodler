---
sidebar_position: 2
---

# [ADVANCED] How Doodler works


Doodler segments images into "classes" that are discrete labels that you define to represent and reduce the dimensionality of features and objects in your imagery.

Each class really represents a spectrum of textures, colors, and spatial extents. It can be difficult to define a good set, but the golden rule (in my humble opinion) is that each class in the set of classes should represent a spectrum of image features that collectively have much greater inter-class variability than intra-class variability.

That can be difficult to define in practice with generality, but here's a tip: imagine a really low- (coarse-) resolution version of your image (or make one!) and ask yourself, "are there are two sets of features that could be easily misconstrued as being in the same class, but are in fact two separate classes?" If the answer in yes, ideally you should lump those classes into a merged class. If your intended outcome dictates that's not possible to do, ask yourself if the two classes could be broken up further, say into 3 or 4 classes, to change your answer to the above question?

In general, doodler is optimized for classes that are relatively large (in terms of contiguous spatial extent) and distinct from one another.

## Image feature extraction and Machine Learning model
The program uses two Machine Learning models in concert to segment images using user-provided annotations or 'doodles'. The two models are 1) Random Forest, or RF, and 2) Fully Connected Conditional Ransom Field (CRF). The program adopts a strategy similar to that described by Buscombe and Ritchie (2018), in that a 'global' model trained on many samples is used to provide an initial segmentation on each sample image, then that initial segmentation is refined by a CRF, which operates on a task specific level. In Buscombe and Ritchie (2018), the model was a deep neural network trained in advance on large numbers of samples and labels. Here, the model is built as we go, building progressively from user inputs. Doodler uses a Random Forest as the baseline global model, and the CRF implementation is the same as that desribed by Buscombe and Ritchie (2018).

Images are labelled in sessions. During a session, a RF model is initialized then built progressively using provided labels from each image. The RF model uses features extracted from the image

    * Gaussian blur over a range of scales
    * Sobel filter of the Gaussian blurred images
    * Matrix of texture values extacted over a range of scales as the 1st eigenvalue of the Hessian Matrix
    * Matrix of texture values extacted over a range of scales as the 2nd eigenvalue of the Hessian Matrix

and the relative pixel locations in x and y are also used as features in RF model fitting and for prediction. Each RF prediction (a label matrix of integer values, each integer corresponding to a unique class). The CRF builds a model for the likelihood of the RF-predicted labels based on the distributions of features it extracts from the imagery, and can reclassify pixels (it is intended to do so). Its feature extraction and decision making behavior is complex and governed by parameters. The user can control the parameter values for CRF and RF models using the graphical user interface.
