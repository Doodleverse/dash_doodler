---
sidebar_position: 1
---

# Installing Doodler on a PC for your own use

These instructions are for regular python and command line users.

Doodler is a python program that is designed to run within a [conda](https://docs.conda.io/en/latest/) environment, accessed from the command line (terminal). It is designed to work on all modern Windows, Mac OS X, and Linux distributions, with python 3.6 or greater. It therefore requires some familiarity with process of navigating to a directory, and running a python script from a command line interface (CLI) such as a Anaconda prompt terminal window, Powershell terminal window, git bash shell, or other terminal/command line interface.

Conda environments are created in various ways, but the following instructions assume you are using the popular (and free) [Anaconda](https://www.anaconda.com/products/individual) python/conda distribution. A good lightweight alternative to Anaconda is [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)

### Install
Open a CLI (see above) - if in doubt, we recommend using the Anaconda shell that is installed by the Anaconda installation process.

![](/img/install/install1.PNG)

#### Download repository from github
Navigate to a suitable place on your PC using shell navigation commands (`cd`) and clone/download the repository

![](/img/install/install2.PNG)


```shell
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git
```

(the `--depth 1` just means it will pull only the latest version of the software, saving time and disk space)

![](/img/install/install3.PNG)

Then change directory (`cd`) to the dash_doodler folder that you just downloaded

```shell
cd dash_doodler
```

#### Install the requirements

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
conda env create --file environment/dashdoodler.yml
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

### The Doodler files and file structure

It's important you know what the various files do. It's also important to stress that all these files are arranged in folders whose precise relative location matters, so please no moving or deleting of files - it will cause you problems

![](/img/install/install4.PNG)

#### Assets folder
This is the `assets` folder where you should put the images you want to label. The program comes with a set of default files that can optionally download by running the `download_data.py` script that look like this:

![](/img/install/install5.PNG)

... before you start labeling your images you should move or delete these default images.

#### Results folder
This is the results folder. When you start "doodling", the program will automatically create a new timestamped folder like this containing your results

![](/img/install/install6.PNG)

#### Labeled folder
This is the 'labeled' folder, where the program copies the images you have labeled to. It does this so it can keep track of which images you have labeled and which remain in the assets folder. It's also sometimes useful for you to see what and how many images you have labeled during a session.

![](/img/install/install9.PNG)


#### Utils folder
Finally, these python scripts are run from the command line, and are a handy collection of things once you have some data, so we'll talk about them later
![](/img/install/install8.PNG)




<!-- A video is available that covers the installation process:

INSERT VIDEO HERE -->

<!--
## Using Doodler



### Run the program
Run the app.

```shell
python doodler.py
``` -->


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
