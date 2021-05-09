---
sidebar_position: 1
---


# Installing Doodler on a PC for your own use

These instructions are for regular python and command line users.

Doodler is a python program that is designed to run within a [conda](https://docs.conda.io/en/latest/) environment, accessed from the command line (terminal). It is designed to work on all modern Windows, Mac OS X, and Linux distributions, with python 3.6 or greater. It therefore requires some familiarity with process of navigating to a directory, and running a python script from a command line interface (CLI) such as a Anaconda prompt terminal window, Powershell terminal window, git bash shell, or other terminal/command line interface.

Conda environments are created in various ways, but the following instructions assume you are using the popular (and free) [Anaconda](https://www.anaconda.com/products/individual) python/conda distribution. A good lightweight alternative to Anaconda is [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)

### Install
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
