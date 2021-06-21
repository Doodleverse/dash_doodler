---
sidebar_position: 2
---


# Case Study 3: NOAA post-hurricane reconnaissance aerial imagery

This is Emergency Response Imagery collected by the National Geodetic Survey Remote Sensing Division of the US National Oceanographic and Atmospheric Administration (NOAA; https://storms.ngs.noaa.gov). Such imagery helps coastal scientists study impacts of hurricanes and other storms on coastal communities, landforms and ecosystems (see, for example, [Goldstein et al. 2020](../tutorial-extras/references) and [Lazarus et al. 2021](../tutorial-extras/references)). The imagery depicts a mixture of open sandy coasts, marshes, and bay/estuarine environments, with various degrees of development. The classes we use below are water (blue), sand (red), marsh (yellow), and development (green).

## Example 1

Image and standardized image
![](/img/casestudy2/ex1/filt.png)

Extracted features
![](/img/casestudy2/ex1/feats.png)

Doodles
![](/img/casestudy2/ex1/doodles.png)

<!-- Feature importances
![](/img/casestudy2/ex1/feat_imps.png) -->

CRF TTA
![](/img/casestudy2/ex1/tta.png)

Looks bad:
![](/img/casestudy2/ex1/label.png)

What went wrong?
![](/img/casestudy2/ex1/crf_spatfilt.png)

RF spatial filter better
![](/img/casestudy2/ex1/rf_spatfilt.png)

RF label image better

![](/img/casestudy2/ex1/rf_label.png)



## Example 2

Image and standardized image
![](/img/casestudy2/ex2/filt.png)

Extracted features
![](/img/casestudy2/ex2/feats.png)

Doodles
![](/img/casestudy2/ex2/doodles.png)
<!--
Feature importances
![](/img/casestudy2/ex2/feat_imps.png) -->

CRF TTA
![](/img/casestudy2/ex2/tta.png)

CRF spatial filter:
![](/img/casestudy2/ex2/crf_spatfilt.png)


Looks ok:
![](/img/casestudy2/ex2/label.png)


RF label image worse

![](/img/casestudy2/ex2/rf_label.png)
