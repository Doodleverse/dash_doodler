/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'Doodler',
  tagline: 'An interactive "human in the loop" Machine Learning program for image segmentation',
  url: 'https://dbuscombe-usgs.github.io/dash_doodler/',
  baseUrl: '/',
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
          label: 'Doodler gitHub page',
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
              label: 'GitHub',
              href: 'https://github.com/dbuscombe-usgs/dash_doodler',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Marda Science, LLC.`,
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
