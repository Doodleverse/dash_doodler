(window.webpackJsonp=window.webpackJsonp||[]).push([[30],{101:function(e,t,r){"use strict";r.r(t),r.d(t,"frontMatter",(function(){return i})),r.d(t,"metadata",(function(){return c})),r.d(t,"toc",(function(){return l})),r.d(t,"default",(function(){return p}));var a=r(3),n=r(7),o=(r(0),r(117)),i={title:"How to split up large photos",author:"Dan Buscombe",authorURL:"http://twitter.com/magic_walnut"},c={permalink:"/dash_doodler/blog/2021/05/09/blog-post",editUrl:"https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/blog/blog/2021-05-09-blog-post.md",source:"@site/blog/2021-05-09-blog-post.md",title:"How to split up large photos",description:"Doodler works well with small to medium sized imagery where the features and objects can be labeled without much or any zoom or pan. This depends a lot on the image resolution and content so it is difficult to make general guidelines.",date:"2021-05-09T00:00:00.000Z",formattedDate:"May 8, 2021",tags:[],readingTime:1.335,truncated:!1,prevItem:{title:"How to use viz_npz.py",permalink:"/dash_doodler/blog/2021/05/15/blog-post"},nextItem:{title:"Splitting up large geoTIFF orthomosaics",permalink:"/dash_doodler/blog/2020/08/01/blog-post"}},l=[{value:"Halves",id:"halves",children:[]},{value:"Quarters",id:"quarters",children:[]},{value:"Specific size",id:"specific-size",children:[]},{value:"Recombine",id:"recombine",children:[]}],s={toc:l};function p(e){var t=e.components,i=Object(n.a)(e,["components"]);return Object(o.b)("wrapper",Object(a.a)({},s,i,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"Doodler works well with small to medium sized imagery where the features and objects can be labeled without much or any zoom or pan. This depends a lot on the image resolution and content so it is difficult to make general guidelines."),Object(o.b)("p",null,"But it's easy enough to chop images into pieces, so you should experiment with a few different image sizes."),Object(o.b)("p",null,"Let's start with this image called ",Object(o.b)("inlineCode",{parentName:"p"},"big.jpg"),":"),Object(o.b)("p",null,Object(o.b)("img",{src:r(133).default})),Object(o.b)("p",null,"I recommend the command-line program ",Object(o.b)("a",{parentName:"p",href:"https://imagemagick.org/index.php"},"imagemagick"),", available for all major platforms. It's an incredibly powerful and useful set of tools for manipulating images. You can use the imagemagick command line ",Object(o.b)("a",{parentName:"p",href:"https://imagemagick.org/script/command-line-processing.php"},"tools")," for splitting and merging imagery. We use the ",Object(o.b)("inlineCode",{parentName:"p"},"magick")," command (",Object(o.b)("inlineCode",{parentName:"p"},"convert")," on some Linux distributions)"),Object(o.b)("h2",{id:"halves"},"Halves"),Object(o.b)("p",null,"Split into two lengthways:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-cmd"},"magick big.jpg -crop 50%x100% +repage fordoodler_%02d.jpg\n")),Object(o.b)("p",null,Object(o.b)("img",{src:r(134).default}),"\n",Object(o.b)("img",{src:r(135).default})),Object(o.b)("h2",{id:"quarters"},"Quarters"),Object(o.b)("p",null,"Following the same logic, to chop the image into quarters, use:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-cmd"},"magick big.jpg -crop 50%x50% +repage quarters_fordoodler_%02d.jpg\n")),Object(o.b)("p",null,"The first two quarters are shown below:"),Object(o.b)("p",null,Object(o.b)("img",{src:r(136).default}),"\n",Object(o.b)("img",{src:r(137).default})),Object(o.b)("h2",{id:"specific-size"},"Specific size"),Object(o.b)("p",null,"To chop the image into tiles of a specific size, for example 1024x1024 pixels, use:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-cmd"},"magick big.jpg -crop 1024x1024 +repage px1024_fordoodler_%02d.jpg\n")),Object(o.b)("p",null,"The first three tiles are shown below:"),Object(o.b)("p",null,Object(o.b)("img",{src:r(138).default}),"\n",Object(o.b)("img",{src:r(139).default}),"\n",Object(o.b)("img",{src:r(140).default})),Object(o.b)("p",null,"Easy peasy!"),Object(o.b)("h2",{id:"recombine"},"Recombine"),Object(o.b)("p",null,"After you've labeled, you may want to recombine your label image. Imagemagick includes the ",Object(o.b)("a",{parentName:"p",href:"https://imagemagick.org/script/montage.php"},"montage")," tool that is handy for the task. For example, the image quarters can be recombined like this:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-cmd"},"magick montage -mode concatenate -tile 2x2 quarters*.jpg recombined.jpg\n")),Object(o.b)("p",null,"and the equivalent command to combine the two vertical halves is:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-cmd"},"magick montage -mode concatenate -tile 2x1 fordoodler*.jpg recombined.jpg\n")),Object(o.b)("p",null,"Happy image cropping!"))}p.isMDXComponent=!0},117:function(e,t,r){"use strict";r.d(t,"a",(function(){return d})),r.d(t,"b",(function(){return m}));var a=r(0),n=r.n(a);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function c(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,a,n=function(e,t){if(null==e)return{};var r,a,n={},o=Object.keys(e);for(a=0;a<o.length;a++)r=o[a],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)r=o[a],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var s=n.a.createContext({}),p=function(e){var t=n.a.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):c(c({},t),e)),r},d=function(e){var t=p(e.components);return n.a.createElement(s.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return n.a.createElement(n.a.Fragment,{},t)}},u=n.a.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,i=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),d=p(r),u=a,m=d["".concat(i,".").concat(u)]||d[u]||b[u]||o;return r?n.a.createElement(m,c(c({ref:t},s),{},{components:r})):n.a.createElement(m,c({ref:t},s))}));function m(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=u;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,i[1]=c;for(var s=2;s<o;s++)i[s]=r[s];return n.a.createElement.apply(null,i)}return n.a.createElement.apply(null,r)}u.displayName="MDXCreateElement"},133:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/big-dd7328cee9ac4b04dda6b21b957d8542.jpg"},134:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/fordoodler_00-910f58238c3dc8eb9b818df4103952bc.jpg"},135:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/fordoodler_01-cd7704a3e7237d61292d562dd93817b4.jpg"},136:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/quarters_fordoodler_00-997c532ef45ef578c10d31e5a944e04e.jpg"},137:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/quarters_fordoodler_01-63042ae6c1af10dd1b5adbda080d3aed.jpg"},138:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/px1024_fordoodler_00-ef87dbb9bd2be790add1655e0811f76d.jpg"},139:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/px1024_fordoodler_01-19b2856271599ae0e3a64a977bc57348.jpg"},140:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/px1024_fordoodler_02-9515f059ed92bb7acaefaf75daeae233.jpg"}}]);