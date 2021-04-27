---
sidebar_position: 5
---

# [ADVANCED] How to contribute

Contributions are welcome, and they are greatly appreciated! Credit will always be given.

To provide small edits or suggested content for existing pages, please use the github issues tab: https://github.com/dbuscombe-usgs/dash_doodler/issues


## Making direct changes

Otherwise, please edit the docs directly. Here's how to set up for local development.

1. Fork the dash_doodler repo on Github  https://github.com/dbuscombe-usgs/dash_doodler

2. Clone your fork locally:

```shell
git clone git@github.com:your_name_here/dash_doodler.git
```

3. Install your local copy into a conda/virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

```shell
cd dash_doodler/
conda env create --file install/dashdoodler.yml
conda activate dashdoodler
```

4. Create a branch for local development:

```shell
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

## Write a blog post

Docusaurus creates a **page for each blog post**, but also a **blog index page**, a **tag system**, an **RSS** feed...


Create a file at `blog/2021-02-28-greetings.md`:

```md title="blog/2021-02-28-greetings.md"
---
slug: greetings
title: Greetings!
author: Dan Buscombe
author_title: Doodler Contributor
author_url: https://github.com/dbuscombe-usgs
tags: [greetings]
---

Congratulations, you have made your first post!

Feel free to play around and edit this post as much you like.
```

A new blog post is now available at `http://localhost:3000/blog/greetings`.


## Edit docs

* To edit the content of the frontpage: `src/components/HomepageFeatures.js`
* To edit tutorial pages, go to the subfolders of `docs/`
* To rename sidebar names, edit the `_category_.json` file in each subfolder in `docs/`
* To edit blog pages, go to the subfolders of `blogs/`
* Start here for instructions on how to use docusaurus: https://docusaurus.io/docs/installation

To check the webpage locally, use

```shell
yarn start
```

The webpage will load at http://localhost:3000/dash_doodler.


## Report Bugs

Report bugs at https://github.com/dbuscombe-usgs/dash_doodler/issues.

Please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.
Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.
Write Documentation

We could always use more documentation, whether as part of the docs, in docstrings, or using this software in blog posts, articles, etc.
Get Started!

## Ready to contribute?


5. Commit your changes and push your branch to GitHub:

```shell
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-forked-repo
```

6. Submit a pull request through the GitHub website.
