---
sidebar_position: 4
---

# Frequently Asked Questions

Please read this page before submitting your Issue on Github

### Why is it taking so long?

In typical order of importance:

1. Use smaller images
2. Make fewer doodles
3. Use a larger RF downsample factor
4. Use a larger CRF downsample factor
5. If possible, use fewer classes
6. If possible, use more and faster CPU cores


### What is the maximum number of classes?

This is also called the 'impossible' question because it entirely depends on the nature and variability of your imagery, and the appearance of features assigned to your specific discrete classes. However, there are a few guidelines we can provide:

1. The minimum number of classes is 2. Theoretically, there is no maximum number of classes
2. The maximum number of allowable classes in the program is 24

### I have pngs or tiffs - how do I convert to jpg?

Use imagemagick, specifically the `convert` command - see [here](https://imagemagick.org/script/convert.php)

### My images are too big. How do I tile them?

See [this](../../blog/2021/05/09/blog-post) for regular imagery, and [this](../../blog/2020/08/01/blog-post) for geotiffs

### Okay, I can use Doodler to make label images. Now what?

The primary purpose of Doodler is to create enough label images that Deep Learning image segmentation workflows become viable. To get started, you may follow this self-guided course made by Dr Daniel Buscombe, called ["ML Mondays"](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1#week-3-supervised-image-segmentation).

### How do I visualize my labels?

See [this](../../blog/2021/05/15/blog-post)

## When should I adjust the parameters?

It is generally considered a good thing if you can get a good result without adjusting hyperparameters, or using high values of blur, model independence and data downsample factors. So, focus on how to annotate well and use the hyperparameters as a last resort

On datasets tested to date, we estimate that approximately half or more of images require the addition of annotations beyond the initial sparse set, and approximately a tenth or less require the removal of annotations or the adjustment of hyperparameters.


## When should I increase the downsample factors?
Downsampling is a memory management strategy so use larger values if the program crashes (which is likely because of insufficient RAM or computer memory). Generally, you want to use as low a value of either as possible to avoid downsampling.


## What do the blur and model independence parameters do?

#### Blur
The CRF feature extractor to extract color image features and map them to classes. These features are engineered, by convolving Gaussian kernels with the imagery. The blur parameter controls the degree of allowable similarity in image features between classes, therefore a value of 1 (the default) only tolerates image features with small differences in intensity being assigned the same class label. That's a lot of detail to put into a one-word name, so we called it the `blur` parameter because it generally controls the sharpness of the class boundaries in the label image. It is also named after [these guys](https://youtu.be/WIS-9_KS1To?t=1217).

#### Model independence
The `model independence` is used to define pairwise potentials used by the model to encourage adjacent pixels to be the same class label. Values greater than 1 weight the pairwise potentials more than the unary potentials, which might be useful when the MLP prediction is poor. In general, larger values of `model independence` tend to give the model greater independence, resulting in the reclassification of more pixels. This is intended behavior: the importance of pairwise potentials becomes much greater than unary potentials, and spatial inconsistencies in feature-label pairings have greater likelihood of being reclassified.

#### Caution
Note that neither effect necessarily improves the result.


## Why are my results poor?
See our forthcoming paper. In the meantime, [this](https://youtu.be/dQw4w9WgXcQ?t=76) short video.

<!-- Stop and ask yourself what you are hoping to achieve. Decide on an 'adequate' accuracy for your test case -->


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
