---
sidebar_position: 2
---

# What to do

The basic sequence is as follows:

1. Decide what classes you want to label
2. Decide what images you want to label in *this* Doodler session (be realistic)
Select an image from the files list on the second tab
3. Launch the program
4. Use the different color pens to assign labels to a few pixels for each class, in each portion of the scene where that class appears
5. Hit 'compute/show segmentation' button and wait for the model to complete
6. The segmentation will now be visible as a color image with a semi-transparent overlay
7. Refine by first adding or removing doodles
8. As a last resort, modify the RF and/or CRF parameters to achieve a better result. Any modifications you make the parameters are saved as the 'default' parameter values for subsequent images
9. Move onto the next image. Results from the previous image are saved automatically
10. Once an image is completed, it is removed from the list of files, and the image is copied into the local folder called `labeled/`. This way you can see what images have already been labeled, and you don't mistakenly label an image twice.

Each of these steps is described in more detail below

### Define classes, make a new `classes.txt`
After setting up that environment, create a new `classes.txt` file that tells the program what classes will be labeled (and what buttons to create). The minimum number of classes is 2. The maximum number of classes allowed by the program is 24.

* The file MUST be called `classes.txt` and must be in the top-level directory
* The file can be created using any text editor (e.g. Notepad, Notepad ++) or IDE (e.g. Atom, VScode, spyder)

Here is how you would use Notepad to edit on Windows:
![](/img/install/install11.PNG)

![](/img/install/install12.PNG)


:::tip Tip

Remember to keep your class names short, so the buttons all fit on the screen! We don't recommend using more than 24 classes

Also, no spaces! Use underscores

:::


:::warning Please

no spaces! Use underscores

:::

### Move the images you want to label to the folder named `assets`
Start small; select a few images that a representative of the larger set of images you wish to eventually segment. Moves those images into the `assets/` folder.

:::tip Tip

If starting this process for the first time, we recommend trialing Doodler with no more than 10 images to begin with.

:::


### Launch the program

Open a terminal (shell) and make sure you are in the `dash_doodler` conda environment you set up

```cmd
(base)me@computer:/home/me/$ conda activate dashdoodler
```


then run the program like this

```cmd
(dashdoodler)me@computer:/home/me/$ python doodler.py
```

![](/img/install/install10.PNG)

### Head to the browser
Doodler is a web application you can access using any web browser.

An IP address where you can view the app in your browser will be displayed in the terminal. Some browsers will launch automatically, while others you may have to manually type (or copy/paste) the IP address into a browser.Open a browser and go to [127.0.0.1:8050](http://127.0.0.1:8050/)

:::tip Tip
http://127.0.0.1 is called 'localhost' and 8050 is the port or socket number being used by Doodler

<!-- Doodler is web application that is accessed on your web browser at http://127.0.0.1:8050/ -->
You may have to hit the refresh button. If, after some time doodling things seem odd or buggy, sometimes a browser refresh will fix those glitches.

Doodler is known to work with Chrome, Firefox, and Edge but will likely work well with any modern browser (not Microsoft Internet Explorer).
:::

<!-- [http://127.0.0.1:8050/](http://127.0.0.1:8050/) -->

And you should see something like this:

![](/img/screenshots/1.png)

Excellent, in no time you'll be doodling like this:

![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/quick-satshore2-x2c.gif)

### Identify yourself and select a file

Go to the file list tab and type an identifier such as your initials or name and hit the `submit` button. This is optional, however it really helps keep track of who made what file when working in groups.

![](/img/screenshots/2.png)

:::tip Tip
The ID field can also be used to append short notes or codes to each classified image. It can be changed any time and the most current value will be appended to the output file names
:::

Select an image from the menu

![](/img/screenshots/3.png)

### Start 'doodling'
In the imagery tab, you'll see that the image you selected has loaded

![](/img/screenshots/4.png)

:::tip Take a moment ...

... to decide on what classes are present and in what order you wish to label them. It may not matter yet what order but as you use the program more you may get a sense of the most efficient way to label the scene, which can depend in part on the order in which the classes are labeled

:::

Select the first color button and start to draw, using a mouse or stylus, on the image where that class appears

![](/img/screenshots/6.png)

Do so for all the classes present

![](/img/screenshots/7.png)

:::tip The Golden Rule

Label all classes that are present, in all regions of the image those classes occur. That means (in order of usual importance):

* Don't leave any classes out: if they are present, label them
* Don't let any regions out: a region is hard to define, but as a rule of thumb you should be imagining laying a 4x2 (if the image is portrait) or 2x4 (for landscapes) grid over the image and labeling in each of the cells, such as:

![](/img/screenshots/landscape.png)

or

![](/img/screenshots/portrait.png)

:::

When you have made your doodles, or at least your 'initial doodles', check `Compute/Show segmentation`. You should see the blue box spin to indicate that the computer is thinking.

![](/img/screenshots/8.png)

:::warning ... about that spinning box

If the box spins very slowly or not at all, that is an indication that you don't have enough RAM. Consider exiting the program by hitting the `Ctrl` and the `C` keys on your keyboard together (or `Ctrl+C`), and trying again later using smaller imagery.

The amount of time depends on a few things, principally how large your imagery is, but also how fast your computer's processor is, and how many classes you have (the more classes, the slower it tends to be). If it is too slow, use smaller imagery. Using small imagery is always a good idea anyway.

:::

When it has completed the segmentation will show as a semi transparent color overlay, and your doodles will also still be visible:

![](/img/screenshots/9.png)

Uncheck the `Compute/Show segmentation` before going to the file list tab to select a new image

![](/img/screenshots/10.png)

Select any image from the menu. It is explained below why not a specific one ...

![](/img/screenshots/11.png)

... it is because for some reason your selection never 'sticks' first time. It is a [known issue](https://github.com/dbuscombe-usgs/dash_doodler/issues/12). Unfortunately you have to select any image, then select the image you want. You can confirm your choice has been made by seeing the name in the 'this image/copy' box below matches the image you selected the second time

![](/img/screenshots/12.png)

If your image consists of only only class, make a small mark and hit `compute/show segmentation`.

![](/img/screenshots/13.png)

If you only make doodles of one class, that tells Doodler to just label every pixel in the scene that class. Easy

![](/img/screenshots/14.png)

Keep going!

![](/img/screenshots/15.png)

![](/img/screenshots/16.png)

Initial segmentation needs a little work near the shoreline

![](/img/screenshots/17.png)

Add more water labels near the shoreline. That fixes the shoreline, but makes a small water hole appear on land:

![](/img/screenshots/18.png)

Finally, some doodles to fill that hole

![](/img/screenshots/19.png)

<!-- ![](/img/screenshots/20.png)

![](/img/screenshots/21.png) -->

:::danger Don't over-doodle!

In Doodler, less is often more. Keep your doodles small and strategic; do not attempt to 'color in' the image, which is almost always a counter-productive strategy, leading to model overfitting

:::

![](/img/screenshots/22.png)


### A word on those default label colors ...

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


### Inspect the results

Results (label images and annotation images) are saved to the `results/` folder, inside a subfolder named using the current date and time. The program creates a subfolder each time it is launched, timestamped. That folder contains your results for a session. The results are in a format called npz, which the numpy compressed array storage format. There are a few good reasons why we use that format, but we'll not go into that here.

To visualize the labels you have made, there is a utility provided

```
cd utils
python viz_npz.py
```

It will show you each labeled image, one at a time. Close the image to allow the program to proceed to the next. If you'd rather see all at once, you can ask the program to print them to file (`-l 1` in the below) and suppress plotting to screen (`-p 0`)

```
python viz_npz.py -p 0 -l 1
```

You can also make really detailed analyses of your doodles by running the provided  `plot_label_generation.py` script, whose syntax is

```
cd utils
python plot_label_generation.py {-m [0 or 1] -p [0 or 1] -l [0 or 1]}
```

where
```
-m: save mode. 0=minimal variables, 1=all variables
-p: print figures to screen. 0=no, 1=yes
-l: print figures to file. 0=no, 1=yes
```

### Ongoing use and maintenance

Occasionally, the software is updated. If you watch the repository on github, you can receive alerts about when this happens.

It is best practice to move your images from `assets` and `labeled` folders, and outputs from `results` and `logs`.

For most users who have not modified the contents from the last `git pull`, to update should be a simple matter of carrying out another

```shell
git pull
```

#### The `git` way
However, when you have changes on your working copy, from command line you can use git stash:

:::warning Unsupported territory
Please do not use the Doodler issues tab to ask questions for how to use `git` - they will be closed. Learn how to use git, or develop a workaround that involves moving your files out of the Doodler folder structure, or working from a copy of the Doodler program structure.
:::

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

### Edit/create `my_defaults.py`

The program will make this file, but if you ever need to create or make this file, here is one way how:
![](/img/install/install13.PNG)
![](/img/install/install14.PNG)
