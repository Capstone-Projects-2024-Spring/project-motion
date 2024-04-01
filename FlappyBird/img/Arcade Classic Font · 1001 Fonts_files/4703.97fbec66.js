/*! For license information please see 4703.97fbec66.js.LICENSE.txt */
(self.webpackChunk_1001fonts=self.webpackChunk_1001fonts||[]).push([[4703],{48596:(e,t,a)=>{"use strict";a.d(t,{cV:()=>c,cp:()=>o});var s=a(11504),n=a(17624);const r=["as","disabled"];function c({tagName:e,disabled:t,href:a,target:s,rel:n,role:r,onClick:c,tabIndex:i=0,type:o}){e||(e=null!=a||null!=s||null!=n?"a":"button");const l={tagName:e};if("button"===e)return[{type:o||"button",disabled:t},l];const u=s=>{(t||"a"===e&&function(e){return!e||"#"===e.trim()}(a))&&s.preventDefault(),t?s.stopPropagation():null==c||c(s)};return"a"===e&&(a||(a="#"),t&&(a=void 0)),[{role:null!=r?r:"button",disabled:void 0,tabIndex:t?void 0:i,href:a,target:"a"===e?s:void 0,"aria-disabled":t||void 0,rel:"a"===e?n:void 0,onClick:u,onKeyDown:e=>{" "===e.key&&(e.preventDefault(),u(e))}},l]}const i=s.forwardRef(((e,t)=>{let{as:a,disabled:s}=e,i=function(e,t){if(null==e)return{};var a,s,n={},r=Object.keys(e);for(s=0;s<r.length;s++)a=r[s],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,r);const[o,{tagName:l}]=c(Object.assign({tagName:a,disabled:s},i));return(0,n.jsx)(l,Object.assign({},i,o,{ref:t}))}));i.displayName="Button";const o=i},85904:function(e,t,a){var s,n;void 0===(n="function"==typeof(s=function(){"use strict";var e={},t="en",a=[],s=new RegExp(/^\w+\: +(.+)$/),n=new RegExp(/^\s*((\{\s*(\-?\d+[\s*,\s*\-?\d+]*)\s*\})|([\[\]])\s*(-Inf|\-?\d+)\s*,\s*(\+?Inf|\-?\d+)\s*([\[\]]))\s?(.+?)$/),r=new RegExp(/^\s*(\{\s*(\-?\d+[\s*,\s*\-?\d+]*)\s*\})|([\[\]])\s*(-Inf|\-?\d+)\s*,\s*(\+?Inf|\-?\d+)\s*([\[\]])/),c={locale:h(),fallback:t,placeHolderPrefix:"%",placeHolderSuffix:"%",defaultDomain:"messages",pluralSeparator:"|",add:function(t,s,n,r){var c=r||this.locale||this.fallback,i=n||this.defaultDomain;return e[c]||(e[c]={}),e[c][i]||(e[c][i]={}),e[c][i][t]=s,!1===d(a,i)&&a.push(i),this},trans:function(e,t,a,s){return i(o(e,a,s,this.locale,this.fallback),t||{})},transChoice:function(e,t,a,s,n){var r=o(e,s,n,this.locale,this.fallback),c=parseInt(t,10);return void 0===(a=a||{}).count&&(a.count=t),void 0===r||isNaN(c)||(r=u(r,c,n||this.locale||this.fallback)),i(r,a)},fromJSON:function(e){if("string"==typeof e&&(e=JSON.parse(e)),e.locale&&(this.locale=e.locale),e.fallback&&(this.fallback=e.fallback),e.defaultDomain&&(this.defaultDomain=e.defaultDomain),e.translations)for(var t in e.translations)for(var a in e.translations[t])for(var s in e.translations[t][a])this.add(s,e.translations[t][a][s],a,t);return this},reset:function(){e={},a=[],this.locale=h()}};function i(e,t){var a,s=c.placeHolderPrefix,n=c.placeHolderSuffix;for(a in t){var r=new RegExp(s+a+n,"g");if(r.test(e)){var i=String(t[a]).replace(new RegExp("\\$","g"),"$$$$");e=e.replace(r,i)}}return e}function o(t,s,n,r,c){var i,o,u,f=n||r||c,p=s,d=f.split("_")[0];if(!(f in e))if(d in e)f=d;else{if(!(c in e))return t;f=c}if(null==p)for(var h=0;h<a.length;h++)if(l(f,a[h],t)||l(d,a[h],t)||l(c,a[h],t)){p=a[h];break}if(l(f,p,t))return e[f][p][t];for(;f.length>2&&(i=f.length,u=(o=f.split(/[\s_]+/))[o.length-1].length,1!==o.length);)if(l(f=f.substring(0,i-(u+1)),p,t))return e[f][p][t];return l(c,p,t)?e[c][p][t]:t}function l(t,a,s){return t in e&&a in e[t]&&s in e[t][a]}function u(e,t,a){var i,o,l=[],u=[],d=e.split(c.pluralSeparator),h=[];for(i=0;i<d.length;i++){var v=d[i];n.test(v)?l[(h=v.match(n))[0]]=h[h.length-1]:s.test(v)?(h=v.match(s),u.push(h[1])):u.push(v)}for(o in l)if(r.test(o))if((h=o.match(r))[1]){var g,b=h[2].split(",");for(g in b)if(t==b[g])return l[o]}else{var m=f(h[4]),k=f(h[5]);if(("["===h[3]?t>=m:t>m)&&("]"===h[6]?t<=k:t<k))return l[o]}return u[p(t,a)]||u[0]||void 0}function f(e){return"-Inf"===e?Number.NEGATIVE_INFINITY:"+Inf"===e||"Inf"===e?Number.POSITIVE_INFINITY:parseInt(e,10)}function p(e,t){var a=t;switch("pt_BR"===a&&(a="xbr"),a.length>3&&(a=a.split("_")[0]),a){case"bo":case"dz":case"id":case"ja":case"jv":case"ka":case"km":case"kn":case"ko":case"ms":case"th":case"tr":case"vi":case"zh":default:return 0;case"af":case"az":case"bn":case"bg":case"ca":case"da":case"de":case"el":case"en":case"eo":case"es":case"et":case"eu":case"fa":case"fi":case"fo":case"fur":case"fy":case"gl":case"gu":case"ha":case"he":case"hu":case"is":case"it":case"ku":case"lb":case"ml":case"mn":case"mr":case"nah":case"nb":case"ne":case"nl":case"nn":case"no":case"om":case"or":case"pa":case"pap":case"ps":case"pt":case"so":case"sq":case"sv":case"sw":case"ta":case"te":case"tk":case"ur":case"zu":return 1==e?0:1;case"am":case"bh":case"fil":case"fr":case"gun":case"hi":case"ln":case"mg":case"nso":case"xbr":case"ti":case"wa":return 0===e||1==e?0:1;case"be":case"bs":case"hr":case"ru":case"sr":case"uk":return e%10==1&&e%100!=11?0:e%10>=2&&e%10<=4&&(e%100<10||e%100>=20)?1:2;case"cs":case"sk":return 1==e?0:e>=2&&e<=4?1:2;case"ga":return 1==e?0:2==e?1:2;case"lt":return e%10==1&&e%100!=11?0:e%10>=2&&(e%100<10||e%100>=20)?1:2;case"sl":return e%100==1?0:e%100==2?1:e%100==3||e%100==4?2:3;case"mk":return e%10==1?0:1;case"mt":return 1==e?0:0===e||e%100>1&&e%100<11?1:e%100>10&&e%100<20?2:3;case"lv":return 0===e?0:e%10==1&&e%100!=11?1:2;case"pl":return 1==e?0:e%10>=2&&e%10<=4&&(e%100<12||e%100>14)?1:2;case"cy":return 1==e?0:2==e?1:8==e||11==e?2:3;case"ro":return 1==e?0:0===e||e%100>0&&e%100<20?1:2;case"ar":return 0===e?0:1==e?1:2==e?2:e>=3&&e<=10?3:e>=11&&e<=99?4:5}}function d(e,t){for(var a=0;a<e.length;a++)if(t===e[a])return!0;return!1}function h(){return"undefined"!=typeof document?document.documentElement.lang.replace("-","_"):t}return c})?s.call(t,a,t,e):s)||(e.exports=n)},98624:(e,t,a)=>{"use strict";a.d(t,{Ky:()=>o,MZ:()=>u,eG:()=>l,kv:()=>f});var s=a(11504);a(17624);const n=["xxl","xl","lg","md","sm","xs"],r=s.createContext({prefixes:{},breakpoints:n,minBreakpoint:"xs"}),{Consumer:c,Provider:i}=r;function o(e,t){const{prefixes:a}=(0,s.useContext)(r);return e||a[t]||t}function l(){const{breakpoints:e}=(0,s.useContext)(r);return e}function u(){const{minBreakpoint:e}=(0,s.useContext)(r);return e}function f(){const{dir:e}=(0,s.useContext)(r);return"rtl"===e}},49208:(e,t)=>{var a;!function(){"use strict";var s={}.hasOwnProperty;function n(){for(var e=[],t=0;t<arguments.length;t++){var a=arguments[t];if(a){var r=typeof a;if("string"===r||"number"===r)e.push(a);else if(Array.isArray(a)){if(a.length){var c=n.apply(null,a);c&&e.push(c)}}else if("object"===r){if(a.toString!==Object.prototype.toString&&!a.toString.toString().includes("[native code]")){e.push(a.toString());continue}for(var i in a)s.call(a,i)&&a[i]&&e.push(i)}}}return e.join(" ")}e.exports?(n.default=n,e.exports=n):void 0===(a=function(){return n}.apply(t,[]))||(e.exports=a)}()},44808:(e,t,a)=>{"use strict";var s=a(11504),n=Symbol.for("react.element"),r=Symbol.for("react.fragment"),c=Object.prototype.hasOwnProperty,i=s.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,o={key:!0,ref:!0,__self:!0,__source:!0};function l(e,t,a){var s,r={},l=null,u=null;for(s in void 0!==a&&(l=""+a),void 0!==t.key&&(l=""+t.key),void 0!==t.ref&&(u=t.ref),t)c.call(t,s)&&!o.hasOwnProperty(s)&&(r[s]=t[s]);if(e&&e.defaultProps)for(s in t=e.defaultProps)void 0===r[s]&&(r[s]=t[s]);return{$$typeof:n,type:e,key:l,ref:u,props:r,_owner:i.current}}t.Fragment=r,t.jsx=l,t.jsxs=l},17624:(e,t,a)=>{"use strict";e.exports=a(44808)}}]);