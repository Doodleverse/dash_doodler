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
        With Doodler, you identify pixels of each of the 'classes' present in the scene
        and it does the rest, using a model to 'auto-complete' any pixel you didn't label.
      </>
    ),
  },
  {
    title: 'For images of natural environments',
    Svg: require('../../static/img/undraw_nature_m5ll.svg').default,
    description: (
      <>
        Doodler will work with any type of imagery, but it is designed primarily
        to work with imagery consisting of natural landforms,
        where land composition, cover, and use, are identifiable as characteristic
        textures and colors.
        <br></br>
        <br></br>
        The model Doodler uses to classify each pixel of the scene, i.e. segment the image,
        is optimized for classifying such natural textures and colors, which can vary considerably for
        any given class. Often the model needs most help (i.e. more doodles) near the boundaries where
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
      labeling for segmentation tasks, using polygons, such as <a href="www.makesense.ai">makesense.ai</a>.
      However, for high-resolution imagery with large spatial footprints and complex scenes,
      such as imagery collected from airplanes or satellites,
      exhaustive labeling using polygonal tools can be very time-consuming and, well, exhausting!
      <br></br>
      <br></br>
      Doodler is for rapid semi-supervised approximate segmentation of such imagery.
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
