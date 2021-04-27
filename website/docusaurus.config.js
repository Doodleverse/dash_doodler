/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'Doodler',
  tagline: 'An interactive "human in the loop" Machine Learning program for image segmentation',
  url: 'https://dbuscombe-usgs.github.io',
  baseUrl: '/dash_doodler/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'dbuscombe-usgs', // Usually your GitHub org/user name.
  projectName: 'dash_doodler', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: '',
      logo: {
        alt: 'My Site Logo',
        src: 'img/doodler-logo.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Tutorials',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/dbuscombe-usgs/dash_doodler',
          label: 'Doodler github page',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Tutorials',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Coast Train',
              href: 'https://github.com/dbuscombe-usgs/coast_train',
            },
            {
              label: 'USGS Remote Sensing Coastal Change',
              href: 'https://www.usgs.gov/centers/pcmsc/science/remote-sensing-coastal-change?qt-science_center_objects=0#qt-science_center_objects',
            },
            {
              label: 'Coastal Image Labeler by Dr Evan Goldstein',
              href: 'https://coastalimagelabeler.science/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Github',
              href: 'https://github.com/dbuscombe-usgs/dash_doodler',
            },
            {
              label: 'Makesense.ai (an alternative way to segment imagery)',
              href: 'https://www.makesense.ai/',
            },
          ],
        },

      ],


      copyright: `Doodler is written and maintained by Daniel Buscombe, Marda Science, LLC, contracted to the U.S. Geological Survey Pacific Coastal and Marine Science Center in Santa Cruz, CA. Doodler development is funded by the U.S. Geological Survey Coastal Hazards Program, and is for the primary usage of U.S. Geological Survey scientists, researchers and affiliated colleagues working on the Hurricane Florence Supplemental Project and other coastal hazards research. Thanks to Jon Warrick, Phil Wernette, Chris Sherwood, Jenna Brown, Andy Ritchie, Jin-Si Over, Christine Kranenburg, and the rest of the Florence Supplemental team; to Evan Goldstein and colleagues at University of North Carolina Greensboro; Leslie Hsu at the USGS Community for Data Integration; and LCDR Brodie Wells, formerly of Naval Postgraduate School, Monterey. Copyright © ${new Date().getFullYear()} Marda Science, LLC. `,

    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
