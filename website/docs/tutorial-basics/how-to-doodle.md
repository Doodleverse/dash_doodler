---
sidebar_position: 1
---

# How to Doodle

Videos of Doodler in use

![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/quick-saturban-x2c.gif)


![](https://raw.githubusercontent.com/dbuscombe-usgs/dash_doodler/main/assets/logos/doodler-demo-2-9-21-short-elwha.gif)




## A visual guide


In each of the graphics below, an image is overlain with a semi-transparent mask, color-coded by class label, as indicated by the colorbar. The image on the left is the output of the model (segmented label image) and the image on the right is the annotations (or doodles) provided by a human.

![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto1.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto2.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto3.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto4.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto5.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto6.svg)
![](https://dbuscombe-usgs.github.io/doodle_labeller/docs/assets/Doodler_howto7.svg)


## How to decide on 'classes'

You can have any number of classes (well, 2 or more, with the two classes case styled as something and other) but note that increasing the number of classes also increases the amount of time it takes to interact with the program, since each class must be either labelled or actively skipped.

Be prepared to experiment with classes - how well do they work on a small subset of your imagery?
Look at the unusual images in your set and and decide if there is a class there you hadn't previously thought of.
If you can't decide whether to lump or split certain classes or sets of classes, and experiment on a small test set


:::tip My tip

Use this awesome feature option

:::


:::danger Take care

This action is dangerous

:::

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

This is <Highlight color="#25c2a0">Docusaurus green</Highlight> !

This is <Highlight color="#1877F2">Facebook blue</Highlight> !
