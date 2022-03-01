(window.webpackJsonp=window.webpackJsonp||[]).push([[57],{128:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return i})),n.d(t,"metadata",(function(){return l})),n.d(t,"toc",(function(){return s})),n.d(t,"default",(function(){return c}));var a=n(3),o=n(7),r=(n(0),n(139)),i={title:"What's new in version 1.2.1? (May 10, 2021)",author:"Dan Buscombe",authorURL:"http://twitter.com/magic_walnut"},l={permalink:"/dash_doodler/blog/2021/05/16/blog-post",editUrl:"https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/blog/blog/2021-05-16-blog-post.md",source:"@site/blog/2021-05-16-blog-post.md",title:"What's new in version 1.2.1? (May 10, 2021)",description:"Existing users of Doodler should be aware of the major changes in version 1.2.1, posted May 10. There are a lot of them, and they mean you will use Doodler a little differently going forward. I hope you like the changes!",date:"2021-05-16T00:00:00.000Z",formattedDate:"May 15, 2021",tags:[],readingTime:3.63,truncated:!1,prevItem:{title:"How to Use Zoom & Pan Tools",permalink:"/dash_doodler/blog/2021/07/08/Doodler-Zoom"},nextItem:{title:"How to use viz_npz.py",permalink:"/dash_doodler/blog/2021/05/15/blog-post"}},s=[{value:"GUI:",id:"gui",children:[]},{value:"I/O:",id:"io",children:[]},{value:"Modeling:",id:"modeling",children:[]},{value:"Other:",id:"other",children:[]}],d={toc:s};function c(e){var t=e.components,n=Object(o.a)(e,["components"]);return Object(r.b)("wrapper",Object(a.a)({},d,n,{components:t,mdxType:"MDXLayout"}),Object(r.b)("p",null,"Existing users of Doodler should be aware of the major changes in version 1.2.1, posted May 10. There are a lot of them, and they mean you will use Doodler a little differently going forward. I hope you like the changes!"),Object(r.b)("p",null,"First, Doodler has a new documentation ",Object(r.b)("a",{parentName:"p",href:"https://dbuscombe-usgs.github.io/dash_doodler"},"website")," (you know this because you are here)."),Object(r.b)("p",null,"There are so many new features in this release they are organized by theme ..."),Object(r.b)("h2",{id:"gui"},"GUI:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"Model independence factor ")," replaces the previous \"color class tolerance\" parameter (the mu parameter in the CRF). Higher numbers allow the model to have greater ability to 'undo' class assignments from the RF model. Typically, you want to trust the RF outputs and want to keep this number small.  "),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"Blur factor"),' replaces the previous "Blurring parameter" (the theta parameter in the CRF). Larger values means more smoothing.'),Object(r.b)("li",{parentName:"ul"},"The CRF controls so ",Object(r.b)("inlineCode",{parentName:"li"},"Blur factor"),", then ",Object(r.b)("inlineCode",{parentName:"li"},"Model independence factor "),", then downsample and finally probability of doodle. These are in the order of likelihood that you will need to tweak."),Object(r.b)("li",{parentName:"ul"},"There is no longer an option to apply a median filter. Its usage was generally disaapointing/problematic and has been replaced with an alternaive method (see below)"),Object(r.b)("li",{parentName:"ul"},"The 'show/compute segmentation' button is now blue, so it stands out a little better"),Object(r.b)("li",{parentName:"ul"},"On the RF controls, there is no longer the option to change the 'sigma range'. SIGMA_MAX is now hard-coded at 16. SIGMA_MIN is 1. Tests reveal insensitivity to these parameters. Keeping them as options presents problems for other areas of the workflow; by enforcing the same range of sigma scales, there are the same number (75) output feature maps used by the RF")),Object(r.b)("h2",{id:"io"},"I/O:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"JPEG files with all extensions (.jpg, .jpeg, or .JPG) are now usable inputs"),Object(r.b)("li",{parentName:"ul"},"greyscale and annotations no longer saved to png file, instead to numpy area (npz compressed), which encodes",Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"'image'' = image"),Object(r.b)("li",{parentName:"ul"},"'label' = one-hot-encoded label array"),Object(r.b)("li",{parentName:"ul"},"'color_doodles' = color 3D or color doodles"),Object(r.b)("li",{parentName:"ul"},"'doodles' = 2D or greyscale doodles"))),Object(r.b)("li",{parentName:"ul"},"the npz file is overwritten, but old arrays are kept, prefixed with '0', and prepended with another '0', such that the more '0's the newer, but the above names without '0's are always the newest. Color images are still produced with time tags."),Object(r.b)("li",{parentName:"ul"},"DEFAULT_CRF_DOWNSAMPLE = 4 by default"),Object(r.b)("li",{parentName:"ul"},"In implementation using ",Object(r.b)("inlineCode",{parentName:"li"},"predict_folder.py"),", the user decides between two modes, saving either default basic outputs (final output label) or the full stack out outputs for debugging or optimizing")),Object(r.b)("h2",{id:"modeling"},"Modeling:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"All input images are now standardized and rescaled ","[0,1]",". This results in better portability of RF models, and is general good practice to deal with large outlier values."),Object(r.b)("li",{parentName:"ul"},"The CRF label is now reszied with no antialiasing, and is inpainted at transition areas between classes"),Object(r.b)("li",{parentName:"ul"},"Decreased maximum number of allowable samples in RF model to 200,000"),Object(r.b)("li",{parentName:"ul"},"Small holes and islands in the one-hot encoded RF and CRF masks are now removed. The threshold area in pixels is 2*W, where W is the width of the image in pixels."),Object(r.b)("li",{parentName:"ul"},"Median filtering is now removed. It is no longer needed, creates problems, extra buttons/complexity. Instead ..."),Object(r.b)("li",{parentName:"ul"},"Implements 'one-hot encoded mask spatial filtering'"),Object(r.b)("li",{parentName:"ul"},"Implements inpainting on regions spatially filtered"),Object(r.b)("li",{parentName:"ul"},"Pen width is used as-is; it is no longer exponentially scaled"),Object(r.b)("li",{parentName:"ul"},"SIGMA_MAX=16; SIGMA_MIN=1. Hardcoded. Easier to manage number of features, which now have to be 75. Also, they make very little difference"),Object(r.b)("li",{parentName:"ul"},"in ",Object(r.b)("inlineCode",{parentName:"li"},"predict_folder"),", extracted features are memory mapped to save RAM")),Object(r.b)("h2",{id:"other"},"Other:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"RF feature extraction is now carried out in parallel, and CRF 'test time augmentation' is now in parallel. This should ",Object(r.b)("em",{parentName:"li"},"speed things up")),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"utils/plot_label_generation.py")," is a new script that plots all the minutae of the steps involved in label generation, making plots and large npz files containing lots of variables I will explain later. By default each image is modeled with its own random forest. Uncomment \"#do_sim = True\" to run in 'chain simulation mode', where the model is updated in a chain, simulating what Doodler does."),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"utils/convert_annotations2npz.py")," is a new script that will convert annotation label images and associated images (created and used respectively by/during a previous incarnation of Doodler)"),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"utils/gen_npz_4_zoo.py")," is a new script that will strip just the image and one-hot encoded label stack image for model training with Zoo")))}c.isMDXComponent=!0},139:function(e,t,n){"use strict";n.d(t,"a",(function(){return u})),n.d(t,"b",(function(){return m}));var a=n(0),o=n.n(a);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,a,o=function(e,t){if(null==e)return{};var n,a,o={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var d=o.a.createContext({}),c=function(e){var t=o.a.useContext(d),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=c(e.components);return o.a.createElement(d.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},b=o.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,i=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),u=c(n),b=a,m=u["".concat(i,".").concat(b)]||u[b]||p[b]||r;return n?o.a.createElement(m,l(l({ref:t},d),{},{components:n})):o.a.createElement(m,l({ref:t},d))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,i=new Array(r);i[0]=b;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var d=2;d<r;d++)i[d]=n[d];return o.a.createElement.apply(null,i)}return o.a.createElement.apply(null,n)}b.displayName="MDXCreateElement"}}]);