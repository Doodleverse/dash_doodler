import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const B = (props) => <Text style={{fontWeight: 'bold'}}>{props.children}</Text>

const FeatureList = [
  {
    title: 'Make a few doodles ... classify entire scenes',
    Svg: require('../../static/img/undraw_working_remotely_jh40.svg').default,
    description: (
      <>
        Identify pixels of each of the 'classes' present in the scene with a mouse/stylus,
        and it does the rest, using a model to 'auto-complete' all the pixels you didn't label.
        <br></br>
        <br></br>
        A class is a discrete label such as 'sky', 'water', 'sand', etc. Auto-completion refers to the process of classifying each pixel in the scene into a class using an automated process, otherwise known as image 'segmentation'.
      </>
    ),
  },
  {
    title: 'For images of natural environments',
    Svg: require('../../static/img/undraw_nature_m5ll.svg').default,
    description: (
      <>
        You can use any type of photos, but it is designed
        to work best with imagery consisting of landscapes (natural environments),
        where the surface composition, cover, and sometimes evidence of human and other animal uses, are identifiable as characteristic
        textures and colors.
        <br></br>
        <br></br>
        Doodler uses Machine Learning to classify each pixel of the scene, i.e. segment the image,
        which is optimized for classifying such natural textures and colors, which can vary considerably for
        any given class. Often this 'model' needs most help (i.e. more doodles) near the boundaries where
        one class transitions to another.
      </>
    ),
  },
  {
    title: 'Generate high volumes of labeled imagery quickly',
    Svg: require('../../static/img/undraw_in_no_time_6igu.svg').default,
    description: (
      <>
      There are many great tools for exhaustive (i.e. whole image) image
      labeling using polygons, such as <a href="www.makesense.ai">makesense.ai</a>.
      However, for high-resolution imagery with large spatial footprints depicting complex natural scenes,
      such as imagery collected from airplanes or satellites,
      exhaustive labeling using polygonal tools can be very time-consuming and, well, exhausting!
      <br></br>
      <br></br>
      Doodler is as an alternative tool for rapid approximate segmentation of images that is semi-, not fully, supervised by you.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
