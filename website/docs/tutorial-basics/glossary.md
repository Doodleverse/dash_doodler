---
sidebar_position: 6
---

# Terminology
(page under construction)

(If you find a phrase or word in these documents that you want better explained, submit an [issue](https://github.com/dbuscombe-usgs/dash_doodler/issues) on github, or better yet, [contribute to the docs yourself](../tutorial-extras/how-to-contribute))

### Image segmentation

![](https://dbuscombe-usgs.github.io/MLMONDAYS/img/seg_ex.png)

The images above show some examples of image segmentation. The vegetation in the images in the far left column are segmented to form a binary mask, where white is vegetation and black is everything else (center column). The segmented images (the right column) shows the original image segmented with the mask.

We can also estimate multiple classes at once. The models we use to do this need to be trained using lots of examples of images and their associated label images. To deal with the large intra-class variability, which is often implied in natural landcovers/uses, we require a powerful machine learning model to carry out the segmentation. We will use another type of deep convolutional neural network, this time configured to be an autoencoder-decoder network based on the U-Net.

This is useful for things like:

* quantifying the spatial extent of objects and features of interest

* quantifying everything in the scene as a unique class, with no features or objects unlabelled


### Random Forest

<!-- Random forests are a non-parametric ensemble discriminative classifier

Ensemble means that it relies on aggregating the results of an ensemble of simpler estimators.
Random forests are an example of an ensemble learner built on decision trees. For this reason we'll start by discussing decision trees themselves.
Decision trees are extremely intuitive ways to classify or label objects: you simply ask a series of questions designed to zero-in on the classification.
However, decision trees suffer badly from overfitting.

The use of multiple overfitting estimators can be combined to reduce the effect of this overfitting — this is what underlies an ensemble method called bagging.
Bagging makes use of an ensemble of parallel estimators, each of which over-fits the data, and averages the results to find a better classification.
An ensemble of randomized decision trees is known as a random forest. -->

### Conditional Random Field


### Test-time augmentation

<!-- # Create a Page

Add **Markdown or React** files to `src/pages` to create a **standalone page**:

- `src/pages/index.js` -> `localhost:3000/`
- `src/pages/foo.md` -> `localhost:3000/foo`
- `src/pages/foo/bar.js` -> `localhost:3000/foo/bar`

## Create your first React Page

Create a file at `src/pages/my-react-page.js`:

```jsx title="src/pages/my-react-page.js"
import React from 'react';
import Layout from '@theme/Layout';

export default function MyReactPage() {
  return (
    <Layout>
      <h1>My React page</h1>
      <p>This is a React page</p>
    </Layout>
  );
}
```

A new page is now available at `http://localhost:3000/my-react-page`.

## Create your first Markdown Page

Create a file at `src/pages/my-markdown-page.md`:

```mdx title="src/pages/my-markdown-page.md"
# My Markdown page

This is a Markdown page
```

A new page is now available at `http://localhost:3000/my-markdown-page`. -->
