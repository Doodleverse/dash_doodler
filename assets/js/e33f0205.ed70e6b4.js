(window.webpackJsonp=window.webpackJsonp||[]).push([[57],{128:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return i})),a.d(t,"metadata",(function(){return l})),a.d(t,"toc",(function(){return s})),a.d(t,"default",(function(){return c}));var n=a(3),r=a(7),o=(a(0),a(139)),i={title:"What's new in version 1.2.1? (May 10, 2021)",author:"Dan Buscombe",authorURL:"http://twitter.com/magic_walnut"},l={permalink:"/dash_doodler/blog/2021/05/16/blog-post",editUrl:"https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/blog/blog/2021-05-16-blog-post.md",source:"@site/blog/2021-05-16-blog-post.md",title:"What's new in version 1.2.1? (May 10, 2021)",description:"Existing users of Doodler should be aware of the major changes in version 1.2.1, posted May 10. There are a lot of them, and they mean you will use Doodler a little differently going forward. I hope you like the changes!",date:"2021-05-16T00:00:00.000Z",formattedDate:"May 15, 2021",tags:[],readingTime:3.63,truncated:!1,prevItem:{title:"How to Use the Erase Tool",permalink:"/dash_doodler/blog/2021/07/08/Doodler-Erase"},nextItem:{title:"How to use viz_npz.py",permalink:"/dash_doodler/blog/2021/05/15/blog-post"}},s=[{value:"GUI:",id:"gui",children:[]},{value:"I/O:",id:"io",children:[]},{value:"Modeling:",id:"modeling",children:[]},{value:"Other:",id:"other",children:[]}],d={toc:s};function c(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(o.b)("wrapper",Object(n.a)({},d,a,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"Existing users of Doodler should be aware of the major changes in version 1.2.1, posted May 10. There are a lot of them, and they mean you will use Doodler a little differently going forward. I hope you like the changes!"),Object(o.b)("p",null,"First, Doodler has a new documentation ",Object(o.b)("a",{parentName:"p",href:"https://dbuscombe-usgs.github.io/dash_doodler"},"website")," (you know this because you are here)."),Object(o.b)("p",null,"There are so many new features in this release they are organized by theme ..."),Object(o.b)("h2",{id:"gui"},"GUI:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},Object(o.b)("inlineCode",{parentName:"li"},"Model independence factor ")," replaces the previous \"color class tolerance\" parameter (the mu parameter in the CRF). Higher numbers allow the model to have greater ability to 'undo' class assignments from the RF model. Typically, you want to trust the RF outputs and want to keep this number small.  "),Object(o.b)("li",{parentName:"ul"},Object(o.b)("inlineCode",{parentName:"li"},"Blur factor"),' replaces the previous "Blurring parameter" (the theta parameter in the CRF). Larger values means more smoothing.'),Object(o.b)("li",{parentName:"ul"},"The CRF controls so ",Object(o.b)("inlineCode",{parentName:"li"},"Blur factor"),", then ",Object(o.b)("inlineCode",{parentName:"li"},"Model independence factor "),", then downsample and finally probability of doodle. These are in the order of likelihood that you will need to tweak."),Object(o.b)("li",{parentName:"ul"},"There is no longer an option to apply a median filter. Its usage was generally disaapointing/problematic and has been replaced with an alternaive method (see below)"),Object(o.b)("li",{parentName:"ul"},"The 'show/compute segmentation' button is now blue, so it stands out a little better"),Object(o.b)("li",{parentName:"ul"},"On the RF controls, there is no longer the option to change the 'sigma range'. SIGMA_MAX is now hard-coded at 16. SIGMA_MIN is 1. Tests reveal insensitivity to these parameters. Keeping them as options presents problems for other areas of the workflow; by enforcing the same range of sigma scales, there are the same number (75) output feature maps used by the RF")),Object(o.b)("h2",{id:"io"},"I/O:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"JPEG files with all extensions (.jpg, .jpeg, or .JPG) are now usable inputs"),Object(o.b)("li",{parentName:"ul"},"greyscale and annotations no longer saved to png file, instead to numpy area (npz compressed), which encodes",Object(o.b)("ul",{parentName:"li"},Object(o.b)("li",{parentName:"ul"},"'image'' = image"),Object(o.b)("li",{parentName:"ul"},"'label' = one-hot-encoded label array"),Object(o.b)("li",{parentName:"ul"},"'color_doodles' = color 3D or color doodles"),Object(o.b)("li",{parentName:"ul"},"'doodles' = 2D or greyscale doodles"))),Object(o.b)("li",{parentName:"ul"},"the npz file is overwritten, but old arrays are kept, prefixed with '0', and prepended with another '0', such that the more '0's the newer, but the above names without '0's are always the newest. Color images are still produced with time tags."),Object(o.b)("li",{parentName:"ul"},"DEFAULT_CRF_DOWNSAMPLE = 4 by default"),Object(o.b)("li",{parentName:"ul"},"In implementation using ",Object(o.b)("inlineCode",{parentName:"li"},"predict_folder.py"),", the user decides between two modes, saving either default basic outputs (final output label) or the full stack out outputs for debugging or optimizing")),Object(o.b)("h2",{id:"modeling"},"Modeling:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"All input images are now standardized and rescaled ","[0,1]",". This results in better portability of RF models, and is general good practice to deal with large outlier values."),Object(o.b)("li",{parentName:"ul"},"The CRF label is now reszied with no antialiasing, and is inpainted at transition areas between classes"),Object(o.b)("li",{parentName:"ul"},"Decreased maximum number of allowable samples in RF model to 200,000"),Object(o.b)("li",{parentName:"ul"},"Small holes and islands in the one-hot encoded RF and CRF masks are now removed. The threshold area in pixels is 2*W, where W is the width of the image in pixels."),Object(o.b)("li",{parentName:"ul"},"Median filtering is now removed. It is no longer needed, creates problems, extra buttons/complexity. Instead ..."),Object(o.b)("li",{parentName:"ul"},"Implements 'one-hot encoded mask spatial filtering'"),Object(o.b)("li",{parentName:"ul"},"Implements inpainting on regions spatially filtered"),Object(o.b)("li",{parentName:"ul"},"Pen width is used as-is; it is no longer exponentially scaled"),Object(o.b)("li",{parentName:"ul"},"SIGMA_MAX=16; SIGMA_MIN=1. Hardcoded. Easier to manage number of features, which now have to be 75. Also, they make very little difference"),Object(o.b)("li",{parentName:"ul"},"in ",Object(o.b)("inlineCode",{parentName:"li"},"predict_folder"),", extracted features are memory mapped to save RAM")),Object(o.b)("h2",{id:"other"},"Other:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"RF feature extraction is now carried out in parallel, and CRF 'test time augmentation' is now in parallel. This should ",Object(o.b)("em",{parentName:"li"},"speed things up")),Object(o.b)("li",{parentName:"ul"},Object(o.b)("inlineCode",{parentName:"li"},"utils/plot_label_generation.py")," is a new script that plots all the minutae of the steps involved in label generation, making plots and large npz files containing lots of variables I will explain later. By default each image is modeled with its own random forest. Uncomment \"#do_sim = True\" to run in 'chain simulation mode', where the model is updated in a chain, simulating what Doodler does."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("inlineCode",{parentName:"li"},"utils/convert_annotations2npz.py")," is a new script that will convert annotation label images and associated images (created and used respectively by/during a previous incarnation of Doodler)"),Object(o.b)("li",{parentName:"ul"},Object(o.b)("inlineCode",{parentName:"li"},"utils/gen_npz_4_zoo.py")," is a new script that will strip just the image and one-hot encoded label stack image for model training with Zoo")))}c.isMDXComponent=!0},139:function(e,t,a){"use strict";a.d(t,"a",(function(){return u})),a.d(t,"b",(function(){return m}));var n=a(0),r=a.n(n);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},o=Object.keys(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)a=o[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var d=r.a.createContext({}),c=function(e){var t=r.a.useContext(d),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},u=function(e){var t=c(e.components);return r.a.createElement(d.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},b=r.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,o=e.originalType,i=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),u=c(a),b=n,m=u["".concat(i,".").concat(b)]||u[b]||p[b]||o;return a?r.a.createElement(m,l(l({ref:t},d),{},{components:a})):r.a.createElement(m,l({ref:t},d))}));function m(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=a.length,i=new Array(o);i[0]=b;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:n,i[1]=l;for(var d=2;d<o;d++)i[d]=a[d];return r.a.createElement.apply(null,i)}return r.a.createElement.apply(null,a)}b.displayName="MDXCreateElement"}}]);