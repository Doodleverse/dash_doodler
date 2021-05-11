---
sidebar_position: 4
---

# Frequently Asked Questions
(page under construction)

Please read this page before submitting your Issue on Github

### Why is it taking so long?

In typical order of importance:

1. Use smaller images
2. Make fewer doodles
3. Use a larger RF downsample factor
4. Use a larger CRF downsample factor
5. If possible, use fewer classes
6. If possible, use more and faster CPU cores

### Why are my results poor?
See [this](../../blog/2021/05/11/blog-post)

<!-- Stop and ask yourself what you are hoping to achieve. Decide on an 'adequate' accuracy for your test case -->

### What is the maximum number of classes?

<!-- This is also called the 'impossible' question because it entirely depends on the nature and variability of your imagery, and the appearance of features assigned to your specific discrete classes.
However, there are a few guidelines we can provide:

1. The minimum number of classes is 2. Theoretically, there is no maximum number of classes
2. -->

### How should I choose classes?

See [this](how-to-doodle#how-to-decide-on-classes)


### Okay, I can use Doodler to make label images. Now what?

The primary purpose of Doodler is to create enough label images that Deep Learning image segmentation workflows become viable. To get started, you may follow this self-guided course made by Dr Daniel Buscombe, called ["ML Mondays"](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1#week-3-supervised-image-segmentation).

### How do I visualize my labels?

See [this](../../blog/2021/05/15/blog-post)


### What do the blur and model independence parameters dp?

See [this](../../blog/2021/05/14/blog-post)

## My images are too big. How to I tile them?

See [this](../../blog/2021/05/09/blog-post) for regular imagery, and [this](../../blog/2020/08/01/blog-post) for geotiffs



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
