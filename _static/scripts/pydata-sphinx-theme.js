(()=>{"use strict";var e,t,n,r,o={803:(e,t,n)=>{n.d(t,{Gu:()=>r,qu:()=>o});const r=e=>"string"==typeof e&&/^[v\d]/.test(e)&&a.test(e),o=(e,t,n)=>{h(n);const r=((e,t)=>{const n=c(e),r=c(t),o=n.pop(),a=r.pop(),i=l(n,r);return 0!==i?i:o&&a?l(o.split("."),a.split(".")):o||a?o?-1:1:0})(e,t);return u[n].includes(r)},a=/^[v^~<>=]*?(\d+)(?:\.([x*]|\d+)(?:\.([x*]|\d+)(?:\.([x*]|\d+))?(?:-([\da-z\-]+(?:\.[\da-z\-]+)*))?(?:\+[\da-z\-]+(?:\.[\da-z\-]+)*)?)?)?$/i,c=e=>{if("string"!=typeof e)throw new TypeError("Invalid argument expected string");const t=e.match(a);if(!t)throw new Error(`Invalid argument not valid semver ('${e}' received)`);return t.shift(),t},i=e=>"*"===e||"x"===e||"X"===e,d=e=>{const t=parseInt(e,10);return isNaN(t)?e:t},s=(e,t)=>{if(i(e)||i(t))return 0;const[n,r]=((e,t)=>typeof e!=typeof t?[String(e),String(t)]:[e,t])(d(e),d(t));return n>r?1:n<r?-1:0},l=(e,t)=>{for(let n=0;n<Math.max(e.length,t.length);n++){const r=s(e[n]||"0",t[n]||"0");if(0!==r)return r}return 0},u={">":[1],">=":[0,1],"=":[0],"<=":[-1,0],"<":[-1]},m=Object.keys(u),h=e=>{if("string"!=typeof e)throw new TypeError("Invalid operator type, expected string but got "+typeof e);if(-1===m.indexOf(e))throw new Error(`Invalid operator, expected one of ${m.join("|")}`)}},375:(e,t,n)=>{function r(e){"loading"!=document.readyState?e():document.addEventListener("DOMContentLoaded",e)}n.d(t,{A:()=>r})},937:(e,t,n)=>{n.a(e,(async(e,t)=>{try{var r=n(375),o=n(803),a=window.matchMedia("(prefers-color-scheme: dark)");function l(e){document.documentElement.dataset.theme=a.matches?"dark":"light"}function u(e){"light"!==e&&"dark"!==e&&"auto"!==e&&(console.error(`Got invalid theme mode: ${e}. Resetting to auto.`),e="auto");var t=a.matches?"dark":"light";document.documentElement.dataset.mode=e;var n="auto"==e?t:e;document.documentElement.dataset.theme=n,document.querySelectorAll(".dropdown-menu").forEach((e=>{"dark"===n?e.classList.add("dropdown-menu-dark"):e.classList.remove("dropdown-menu-dark")})),localStorage.setItem("mode",e),localStorage.setItem("theme",n),console.log(`[PST]: Changed to ${e} mode using the ${n} theme.`),a.onchange="auto"==e?l:""}function m(){const e=document.documentElement.dataset.defaultMode||"auto",t=localStorage.getItem("mode")||e;var n,r;u(((r=(n=a.matches?["auto","light","dark"]:["auto","dark","light"]).indexOf(t)+1)===n.length&&(r=0),n[r]))}function h(){u(document.documentElement.dataset.mode),document.querySelectorAll(".theme-switch-button").forEach((e=>{e.addEventListener("click",m)}))}function p(){window.addEventListener("activate.bs.scrollspy",(function(){document.querySelectorAll(".bd-toc-nav a").forEach((e=>{e.parentElement.classList.remove("active")})),document.querySelectorAll(".bd-toc-nav a.active").forEach((e=>{e.parentElement.classList.add("active")}))}))}function f(){if(!document.querySelector(".bd-docs-nav"))return;var e=document.querySelector("div.bd-sidebar");let t=parseInt(sessionStorage.getItem("sidebar-scroll-top"),10);if(isNaN(t)){var n=document.querySelector(".bd-docs-nav").querySelectorAll(".active");if(n.length>0){var r=n[n.length-1],o=r.getBoundingClientRect().y-e.getBoundingClientRect().y;if(r.getBoundingClientRect().y>.5*window.innerHeight){let t=.25;e.scrollTop=o-e.clientHeight*t,console.log("[PST]: Scrolled sidebar using last active link...")}}}else e.scrollTop=t,console.log("[PST]: Scrolled sidebar using stored browser position...");window.addEventListener("beforeunload",(()=>{sessionStorage.setItem("sidebar-scroll-top",e.scrollTop)}))}var c=()=>{let e=document.querySelectorAll("form.bd-search");return e.length?(1==e.length?e[0]:document.querySelector("div:not(.search-button__search-container) > form.bd-search")).querySelector("input"):void 0},i=()=>{let e=c(),t=document.querySelector(".search-button__wrapper");e===t.querySelector("input")&&t.classList.toggle("show"),document.activeElement===e?e.blur():(e.focus(),e.select(),e.scrollIntoView({block:"center"}))},d=0===navigator.platform.indexOf("Mac")||"iPhone"===navigator.platform;async function v(e){e.preventDefault();let t=`${DOCUMENTATION_OPTIONS.pagename}.html`,n=e.currentTarget.getAttribute("href"),r=n.replace(t,"");try{(await fetch(n,{method:"HEAD"})).ok?location.href=n:location.href=r}catch(e){location.href=r}}async function g(e){try{var t=new URL(e)}catch(n){if(!(n instanceof TypeError))throw n;{const n=await fetch(window.location.origin,{method:"HEAD"});t=new URL(e,n.url)}}const n=await fetch(t);return await n.json()}function y(e,t){const n=`${DOCUMENTATION_OPTIONS.pagename}.html`;t.forEach((e=>{e.dataset.activeVersionName="",e.dataset.activeVersion=""}));const r=(e=e.map((e=>(e.match=e.version==DOCUMENTATION_OPTIONS.theme_switcher_version_match,e.preferred=e.preferred||!1,"name"in e||(e.name=e.version),e)))).map((e=>e.preferred&&e.match)).some(Boolean);var o=!1;e.forEach((e=>{const a=document.createElement("a");a.setAttribute("class","dropdown-item list-group-item list-group-item-action py-1"),a.setAttribute("href",`${e.url}${n}`),a.setAttribute("role","option");const c=document.createElement("span");c.textContent=`${e.name}`,a.appendChild(c),a.dataset.versionName=e.name,a.dataset.version=e.version;let i=r&&e.preferred,d=!r&&!o&&e.match;(i||d)&&(a.classList.add("active"),t.forEach((t=>{t.innerText=e.name,t.dataset.activeVersionName=e.name,t.dataset.activeVersion=e.version})),o=!0),document.querySelectorAll(".version-switcher__menu").forEach((e=>{let t=a.cloneNode(!0);t.onclick=v,e.append(t)}))}))}function b(e){var t=DOCUMENTATION_OPTIONS.VERSION,n=e.filter((e=>e.preferred));if(1!==n.length){const e=0==n.length?"No":"Multiple";return void console.log(`[PST] ${e} versions marked "preferred" found in versions JSON, ignoring.`)}const r=n[0].version,a=n[0].url,c=(0,o.Gu)(t)&&(0,o.Gu)(r);if(c&&(0,o.qu)(t,r,"="))return;var i=document.createElement("aside");i.setAttribute("aria-label","Version warning");const d=document.createElement("div"),s=document.createElement("div"),l=document.createElement("strong"),u=document.createElement("a");i.classList="bd-header-version-warning container-fluid",d.classList="bd-header-announcement__content",s.classList="sidebar-message",u.classList="sd-btn sd-btn-danger sd-shadow-sm sd-text-wrap font-weight-bold ms-3 my-1 align-baseline",u.href=`${a}${DOCUMENTATION_OPTIONS.pagename}.html`,u.innerText="Switch to stable version",u.onclick=v,s.innerText="This is documentation for ";const m=t.includes("dev")||t.includes("rc")||t.includes("pre"),h=c&&(0,o.qu)(t,r,">");m||h?l.innerText="an unstable development version":c&&(0,o.qu)(t,r,"<")?l.innerText=`an old version (${t})`:l.innerText=t?`version ${t}`:"an unknown version",i.appendChild(d),d.appendChild(s),s.appendChild(l),s.appendChild(document.createTextNode(".")),s.appendChild(u),document.getElementById("pst-skip-link").after(i)}function w(){new MutationObserver(((e,t)=>{e.forEach((e=>{0!==e.addedNodes.length&&void 0!==e.addedNodes[0].data&&-1!=e.addedNodes[0].data.search("Inserted RTD Footer")&&e.addedNodes.forEach((e=>{document.getElementById("rtd-footer-container").append(e)}))}))})).observe(document.body,{childList:!0})}var s=document.querySelectorAll(".version-switcher__button");const E=s.length>0,_=DOCUMENTATION_OPTIONS.hasOwnProperty("theme_switcher_json_url"),S=DOCUMENTATION_OPTIONS.show_version_warning_banner;if(_&&(E||S)){const O=await g(DOCUMENTATION_OPTIONS.theme_switcher_json_url);y(O,s),S&&b(O)}function T(){document.querySelector(".bd-sidebar-primary [id^=pst-nav-more-links]").classList.add("show")}(0,r.A)(h),(0,r.A)(f),(0,r.A)(p),(0,r.A)((()=>{(()=>{let e=document.querySelectorAll(".search-button__kbd-shortcut");d&&e.forEach((e=>e.querySelector("kbd.kbd-shortcut__modifier").innerText="⌘"))})(),window.addEventListener("keydown",(e=>{let t=c();e.shiftKey||e.altKey||(d?!e.metaKey||e.ctrlKey:e.metaKey||!e.ctrlKey)||!/k/i.test(e.key)?document.activeElement===t&&/Escape/i.test(e.key)&&i():(e.preventDefault(),i())}),!0),document.querySelectorAll(".search-button__button").forEach((e=>{e.onclick=i}));let e=document.querySelector(".search-button__overlay");e&&(e.onclick=i)})),(0,r.A)(w),(0,r.A)(T),t()}catch(N){t(N)}}),1)}},a={};function c(e){var t=a[e];if(void 0!==t)return t.exports;var n=a[e]={exports:{}};return o[e](n,n.exports,c),n.exports}e="function"==typeof Symbol?Symbol("webpack queues"):"__webpack_queues__",t="function"==typeof Symbol?Symbol("webpack exports"):"__webpack_exports__",n="function"==typeof Symbol?Symbol("webpack error"):"__webpack_error__",r=e=>{e&&e.d<1&&(e.d=1,e.forEach((e=>e.r--)),e.forEach((e=>e.r--?e.r++:e())))},c.a=(o,a,c)=>{var i;c&&((i=[]).d=-1);var d,s,l,u=new Set,m=o.exports,h=new Promise(((e,t)=>{l=t,s=e}));h[t]=m,h[e]=e=>(i&&e(i),u.forEach(e),h.catch((e=>{}))),o.exports=h,a((o=>{var a;d=(o=>o.map((o=>{if(null!==o&&"object"==typeof o){if(o[e])return o;if(o.then){var a=[];a.d=0,o.then((e=>{c[t]=e,r(a)}),(e=>{c[n]=e,r(a)}));var c={};return c[e]=e=>e(a),c}}var i={};return i[e]=e=>{},i[t]=o,i})))(o);var c=()=>d.map((e=>{if(e[n])throw e[n];return e[t]})),s=new Promise((t=>{(a=()=>t(c)).r=0;var n=e=>e!==i&&!u.has(e)&&(u.add(e),e&&!e.d&&(a.r++,e.push(a)));d.map((t=>t[e](n)))}));return a.r?s:c()}),(e=>(e?l(h[n]=e):s(m),r(i)))),i&&i.d<0&&(i.d=0)},c.d=(e,t)=>{for(var n in t)c.o(t,n)&&!c.o(e,n)&&Object.defineProperty(e,n,{enumerable:!0,get:t[n]})},c.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),c(937)})();
//# sourceMappingURL=pydata-sphinx-theme.js.map