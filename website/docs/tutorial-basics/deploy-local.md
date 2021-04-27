---
sidebar_position: 2
---


# Running Doodler on a PC for your own use

These brief instructions are for regular python and command line users.

Doodler is a python program that is designed to run within a conda environment, accessed from the command line (terminal). It is designed to work on all modern Windows, Mac OS X, and Linux distributions, with python 3.6 or greater. It therefore requires some familiarity with process of navigating to a directory, and running a python script from a command line interface such as a Anaconda prompt terminal window, Powershell terminal window, git bash shell, or other terminal/command line interface.

The following instructions assume you are using the popular (and free) Anaconda or Miniconda python distribution.

## Install
Clone/download this repository

```shell
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git

cd dash_doodler
```

Install the requirements

:::tip Tip

If you are a regular conda user, now would be a good time to

```shell
conda clean --all
conda update conda
conda update anaconda
```

:::


```shell
conda env create --file install/dashdoodler.yml
conda activate dashdoodler
```

:::danger Tip

If (and only if) the above doesn't work, try this:

```shell
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r install/requirements.txt
```
:::



:::tip Tip

If you are a Windows user (only) who wishes to use unix style commands, install m2-base

```shell
conda install m2-base
```
:::


A video is available that covers the installation process:

INSERT VIDEO HERE


## Use


After setting up that environment, create a classes.txt file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. The maximum number of classes allowed is 24. The images that you upload will go into the assets/ folder. The labels images you create are written to the results folder.

Move your images into the assets folder. For the moment, they must be jpegs with the .jpg extension. Support for other image types forthcoming ...

Run the app. An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Tested so far with Chrome, Firefox, and Edge.

```shell
python doodler.py
```

Open a browser and go to 127.0.0.1:8050. You may have to hit the refresh button. If, after some time doodling things seem odd or buggy, sometimes a browser refresh will fix those glitches.

:::tip My tip

Doodler is web application that is accessed on your web browser at 127.0.0.1:8050

:::

Results (label images and annotation images) are saved to the results/ folder. The program creates a subfolder each time it is launched, timestamped. That folder contains your results images for a session.

The default colormap is plotly's G10, found here. The hex (rgb) color sequence is:

```shell
    #3366CC (51, 102, 204)
    #DC3912 (220, 57, 18)
    #FF9900 (255, 153, 0)
    #109618 (16, 150, 24)
    #990099 (153, 0, 153)
    #0099C6 (0, 153, 198)
    #DD4477 (221, 68, 119)
    #66AA00 (102, 170, 0)
    #B82E2E (184, 46, 46)
    #316395 (49, 99, 149)
```
(you can google search those hex codes and get a color picker view). If you have more than 10 classes, the program uses Light24 instead. This will give you up to 24 classes. Remember to keep your class names short, so the buttons all fit on the screen!

## Ongoing use and maintenance

Occasionally, the software is updated. If you watch the repository on github, you can receive alerts about when this happens.

It is best practice to move your images from `assets` and `labeled` folders, and outputs from `results` and `logs`.

For most users who have not modified the contents from the last `git pull`, to update should be a simple matter of carrying out another

```shell
git pull
```

However, when you have changes on your working copy, from command line use git stash:

```shell
git stash
```

If you need to see what is in your stash, use:

```shell
git stash list
```

This will stash your changes and clear your status report. Next do:

```shell
git pull
```

This will apply stashed changes back to working copy and remove the changes from stash unless you have conflicts. In the case of conflict, they will stay in stash so you can start over if needed:

```shell
git stash pop
```

A one-liner with no stash checking is:

```shell
git stash && git pull && git stash pop
```

Please do not use the Doodler issues tab to ask questions for how to use `git` - they will be closed. Learn how to use git, or move your files out of the Doodler folder structure.


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
