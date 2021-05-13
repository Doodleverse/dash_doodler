(window.webpackJsonp=window.webpackJsonp||[]).push([[7],{126:function(e,t,n){"use strict";n.d(t,"a",(function(){return b})),n.d(t,"b",(function(){return m}));var r=n(0),o=n.n(r);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=o.a.createContext({}),p=function(e){var t=o.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},b=function(e){var t=p(e.components);return o.a.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},d=o.a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,a=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),b=p(n),d=r,m=b["".concat(a,".").concat(d)]||b[d]||u[d]||i;return n?o.a.createElement(m,l(l({ref:t},s),{},{components:n})):o.a.createElement(m,l({ref:t},s))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,a=new Array(i);a[0]=d;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:r,a[1]=l;for(var s=2;s<i;s++)a[s]=n[s];return o.a.createElement.apply(null,a)}return o.a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},73:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return a})),n.d(t,"metadata",(function(){return l})),n.d(t,"toc",(function(){return c})),n.d(t,"default",(function(){return p}));var r=n(3),o=n(7),i=(n(0),n(126)),a={title:"Splitting up large geoTIFF orthomosaics",author:"Dan Buscombe",authorURL:"http://twitter.com/magic_walnut"},l={permalink:"/dash_doodler/blog/2020/08/01/blog-post",editUrl:"https://github.com/dbuscombe-usgs/dash_doodler/edit/master/website/blog/blog/2020-08-01-blog-post.md",source:"@site/blog/2020-08-01-blog-post.md",title:"Splitting up large geoTIFF orthomosaics",description:"Doodler can work with really large images, but it is usually best to keep your images < 10,000 pixels in any dimension, because then the program will do CRF inference on the whole image at once rather than in chunks. This usually results in better image segmentations that are more consistent with your doodles.",date:"2020-08-01T00:00:00.000Z",formattedDate:"July 31, 2020",tags:[],readingTime:1.53,truncated:!1,prevItem:{title:"How to split up large photos",permalink:"/dash_doodler/blog/2021/05/09/blog-post"},nextItem:{title:"merge a 3-band and 1-band image",permalink:"/dash_doodler/blog/2020/07/31/blog-post"}},c=[],s={toc:c};function p(e){var t=e.components,n=Object(o.a)(e,["components"]);return Object(i.b)("wrapper",Object(r.a)({},s,n,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"Doodler")," can work with really large images, but it is usually best to keep your images < 10,000 pixels in any dimension, because then the program will do CRF inference on the whole image at once rather than in chunks. This usually results in better image segmentations that are more consistent with your doodles."),Object(i.b)("p",null,"So, this post is all about how you make smaller image tiles from a very large geoTIFF format orthomosaic, using python. The smaller tiles will also be written out as image tiles, with their relative position in the larger image described in the file name, for easy reassembly"),Object(i.b)("p",null,"We'll need a dependency not included in the ",Object(i.b)("inlineCode",{parentName:"p"},"doodler")," environment: ",Object(i.b)("inlineCode",{parentName:"p"},"gdal")),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"conda install gdal")),Object(i.b)("p",null,"Now, in python:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"import os, gdal\nfrom gdalconst import *\nfrom glob import glob\n")),Object(i.b)("p",null,"How large do you want your output (square) image tiles to be? (in pixels)"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"tilesize = 5000\n")),Object(i.b)("p",null,"What images would you like to chop up?"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"bigfiles = [\n'Sandwich/2017-01-09_Sandwich_5cm_ortho.tif',\n'Sandwich/2017-02-14_Sandwich_5cm_ortho.tif',\n'Sandwich/2017-03-16_Sandwich_5cm_ortho.tif',\n'Sandwich/2018-01-10_Sandwich_5cm_ortho.tif',\n]\n")),Object(i.b)("p",null,"List the widths and heights of those input ",Object(i.b)("inlineCode",{parentName:"p"},"bigfiles")),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"widths = [13314, 13314, 13314, 19972]\nheights = [6212, 6212, 6212, 9319]\n")),Object(i.b)("p",null,"Specify a new folder for each set of image tiles (one per big image)"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"folders = ['Sandwich/2017-01-09_5cm', 'Sandwich/2017-02-14_5cm',\\\n          'Sandwich/2017-03-16_5cm','Sandwich/2017-01-10_5cm']\n")),Object(i.b)("p",null,"Make file name prefixes by borrowing the folder name:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"prefixes = [f.split('/')[-1] for f in folders]\n")),Object(i.b)("p",null,"Finally, loop through each file, chop it into chunks using ",Object(i.b)("inlineCode",{parentName:"p"},"gdal_translate"),", called by an ",Object(i.b)("inlineCode",{parentName:"p"},"os.system()")," command. Then moves the tiles into their respective folders"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},'for b,f,p in zip(bigfiles, folders, prefixes):\n\n    # chop the image into chunks\n    for i in range(0, widths[k], tilesize):\n        for j in range(0, heights[k], tilesize):\n            gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(tilesize)+", " \\\n                +str(tilesize)+" "+b+" "+p+"_"+str(i)+"_"+str(j)+".tif"\n            os.system(gdaltranString)\n\n    ##move those chunks to a directory\n    os.mkdir(f)\n    os.system(\'mv \'+p+\'*.tif \'+f)\n')))}p.isMDXComponent=!0}}]);