---
sidebar_position: 2
---


# Running Doodler on a PC for your own use

These instructions are for regular python and command line users.

Doodler is a python program that is designed to run within a [conda](https://docs.conda.io/en/latest/) environment, accessed from the command line (terminal). It is designed to work on all modern Windows, Mac OS X, and Linux distributions, with python 3.6 or greater. It therefore requires some familiarity with process of navigating to a directory, and running a python script from a command line interface (CLI) such as a Anaconda prompt terminal window, Powershell terminal window, git bash shell, or other terminal/command line interface.

Conda environments are created in various ways, but the following instructions assume you are using the popular (and free) [Anaconda](https://www.anaconda.com/products/individual) python/conda distribution. A good lightweight alternative to Anaconda is [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)

## Install
Open a CLI (see above) - if in doubt, we recommend using the Anaconda shell that is installed by the Anaconda installation process.


Navigate to a suitable place on your PC using shell navigation commands (`cd`) and clone/download the repository

```shell
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git
```

(the `--depth 1` just means it will pull only the latest version of the software, saving time and disk space)

Then change directory (`cd`) to the dash_doodler folder that you just downloaded

```shell
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

Next create a new conda evironment, called `dashdoodler`

```shell
conda env create --file install/dashdoodler.yml
```

We are using packages from [conda-forge](https://anaconda.org/conda-forge/conda), a channel of software versions contributed by the community

:::danger Workaround

If (and only if) the above doesn't work, try this:

```shell
conda create --name dashdoodler python=3.6
conda activate dashdoodler
conda install -c conda-forge pydensecrf cairo
pip install -r install/requirements.txt
```
:::

Finally, activate the environment so you can use it

```shell
conda activate dashdoodler
```

:::tip Tip

If you are a Windows user (only) who wishes to use unix style commands, additionally install `m2-base`

```shell
conda install m2-base
```
:::


A video is available that covers the installation process:

INSERT VIDEO HERE


## Using Doodler

### Define classes, make a new `classes.txt`
After setting up that environment, create a new `classes.txt` file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. The maximum number of classes allowed by the program is 24.

* The file MUST be called `classes.txt` and must be in the top-level directory
* The file can be created using any text editor (e.g. Notepad, Notepad ++) or IDE (e.g. Atom, VScode, spyder)

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
(you can google search those hex codes and get a color picker view).

export const Highlight = ({children, color}) => (
  <span
    style={{
      backgroundColor: color,
      borderRadius: '20px',
      color: '#fff',
      padding: '10px',
      cursor: 'pointer',
    }}
    onClick={() => {
      alert(`You clicked the color ${color} with label ${children}`)
    }}>
    {children}
  </span>
);


This is <Highlight color="#3366CC">class 1</Highlight>, <Highlight color="#DC3912">class 2</Highlight>, <Highlight color="#B82E2E">class 9</Highlight>, etc


If you have more than 10 classes, the program uses Light24 instead. This will give you up to 24 classes, which are (hex / RGB)

```shell
    #FD3216 (253,50,22)
    #00FE35 (0,254,53)
    #6A76FC (106,118,252)
    #FED4C4 (254,212,196)
    #FE00CE (254,0,206)
    #0DF9FF (13,249,255)
    #F6F926 (246,249,38)
    #FF9616 (255,150,22)
    #479B55 (71,155,85)
    #EEA6FB (238,166,251)
    #DC587D (220,88,125)
    #D626FF (214,38,255)
    #6E899C (110,137,156)
    #00B5F7 (0,181,247)
    #B68E00 (182,142,0)
    #C9FBE5 (201,251,229)
    #FF0092 (255,0,146)
    #22FFA7 (34,255,167)
    #E3EE9E (227,238,158)
    #86CE00 (134,206,0)
    #BC7196 (188,113,150)
    #7E7DCD (126,125,205)
    #FC6955 (252,105,85)
    #E48F72 (228,143,114)
```

In that colorscheme, this is <Highlight color="#FD3216">class 1</Highlight>, <Highlight color="#D626FF">class 12</Highlight>, <Highlight color="#E48F72">class 24</Highlight>


:::tip Tip

Remember to keep your class names short, so the buttons all fit on the screen! We don't recommend using more than 24 classes

:::



### Select images to label and move them to `assets`
Start small; select a few images that a representative of the larger set of images you wish to eventually segment. Moves those images into the `assets/` folder. If starting this process for the first time, we recommend trialing Doodler with no more than 10 images to begin with.


### Run the program
Run the app.

```shell
python doodler.py
```

An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser. Doodler is known to work with Chrome, Firefox, and Edge but will likely work well with any modern browser (not Microsoft Internet Explorer). Open a browser and go to [127.0.0.1:8050](http://127.0.0.1:3000/)

:::tip Tip

Doodler is web application that is accessed on your web browser at http://127.0.0.1:3000/

(http://127.0.0.1 is called 'localhost' and 3000 is the port or socket number being used by Doodler)

You may have to hit the refresh button. If, after some time doodling things seem odd or buggy, sometimes a browser refresh will fix those glitches.

:::


Doodle!

![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/quick-satshore2-x2c.gif)

1. Select an image from the files list on the second tab
2. Use the different color pens to assign labels to a few pixels for each class, in each portion of the scene where that class appears
3. Hit 'compute/show segmentation' button and wait for the model to complete
4. The segmentation will now be visible as a color image with a semi-transparent overlay
5. Refine by first adding or removing doodles
6. As a last resort, modify the RF and/or CRF parameters to achieve a better result. Any modifications you make the parameters are saved as the 'default' parameter values for subsequent images
7. Move onto the next image. Results from the previous image are saved automatically
8. Once an image is completed, it is removed from the list of files, and the image is copied into the local folder called `labeled/`. This way you can see what images have already been labeled, and you don't mistakenly label an image twice.


:::danger Don't over-doodle!

In Doodler, less is often more. Keep your doodles small and strategic; do not attempt to 'color in' the image, which is almost always a counter-productive strategy, leading to model overfitting

:::


### Inspect the results

Results (label images and annotation images) are saved to the `results/` folder, inside a subfolder named using the current date and time. The program creates a subfolder each time it is launched, timestamped. That folder contains your results images for a session.





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
