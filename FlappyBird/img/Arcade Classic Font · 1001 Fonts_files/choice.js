'use strict';(function(){var cmpFile='noModule'in HTMLScriptElement.prototype?'cmp2.js':'cmp2-polyfilled.js';(function(){var cmpScriptElement=document.createElement('script');var firstScript=document.getElementsByTagName('script')[0];cmpScriptElement.async=true;cmpScriptElement.type='text/javascript';var cmpUrl;var tagUrl=document.currentScript.src;cmpUrl='https://cmp.inmobi.com/tcfv2/CMP_FILE?referer=1001fonts.com'.replace('CMP_FILE',cmpFile);cmpScriptElement.src=cmpUrl;firstScript.parentNode.insertBefore(cmpScriptElement,firstScript);})();(function(){var css=""
+" .qc-cmp-button.qc-cmp-secondary-button:hover { "
+"   background-color: #368bd6 !important; "
+"   border-color: transparent !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button:hover { "
+"   color: #ffffff !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button { "
+"   color: #368bd6 !important; "
+" } "
+" .qc-cmp-button.qc-cmp-secondary-button { "
+"   background-color: #eee !important; "
+"   border-color: transparent !important; "
+" } "
+""
+"";var stylesElement=document.createElement('style');var re=new RegExp('&quote;','g');css=css.replace(re,'"');stylesElement.type='text/css';if(stylesElement.styleSheet){stylesElement.styleSheet.cssText=css;}else{stylesElement.appendChild(document.createTextNode(css));}
var head=document.head||document.getElementsByTagName('head')[0];head.appendChild(stylesElement);})();var autoDetectedLanguage='en';var gvlVersion=3;function splitLang(lang){return lang.length>2?lang.split('-')[0]:lang;};function isSupported(lang){var langs=['en','fr','de','it','es','da','nl','el','hu','pt','pt-br','pt-pt','ro','fi','pl','sk','sv','no','ru','bg','ca','cs','et','hr','lt','lv','mt','sl','tr','zh'];return langs.indexOf(lang)===-1?false:true;};if(gvlVersion===2&&isSupported(splitLang(document.documentElement.lang))){autoDetectedLanguage=splitLang(document.documentElement.lang);}else if(gvlVersion===3&&isSupported(document.documentElement.lang)){autoDetectedLanguage=document.documentElement.lang;}else if(isSupported(splitLang(navigator.language))){autoDetectedLanguage=splitLang(navigator.language);};var choiceMilliSeconds=(new Date).getTime();window.__tcfapi('init',2,function(){},{"coreConfig":{"inmobiAccountId":"YWGwxFBETd3dz","privacyMode":["GDPR"],"hashCode":"9I+aIXlhuS8pzaGjJu9URA","publisherCountryCode":"DE","publisherName":"1001 Fonts","vendorPurposeIds":[1,2,7,8,10,11,3,5,4,6,9],"vendorFeaturesIds":[1,2,3],"vendorPurposeLegitimateInterestIds":[7,8,9,2,10,11],"vendorSpecialFeaturesIds":[2,1],"vendorSpecialPurposesIds":[1,2],"googleEnabled":true,"consentScope":"service","thirdPartyStorageType":"iframe","consentOnSafari":false,"displayUi":"inEU","defaultToggleValue":"off","initScreenRejectButtonShowing":true,"initScreenCloseButtonShowing":false,"softOptInEnabled":false,"showSummaryView":true,"persistentConsentLinkLocation":3,"displayPersistentConsentLink":false,"uiLayout":"popup","publisherLogo":"https://st.1001fonts.net/img/1001fonts-logo.svg?qc-size=300,108","vendorListUpdateFreq":365,"publisherPurposeIds":[],"initScreenBodyTextOption":1,"publisherConsentRestrictionIds":[],"publisherLIRestrictionIds":[],"publisherPurposeLegitimateInterestIds":[],"publisherSpecialPurposesIds":[],"publisherFeaturesIds":[],"publisherSpecialFeaturesIds":[],"stacks":[1,42],"lang_":autoDetectedLanguage,"gvlVersion":3,"totalVendors":804,"gbcConfig":{"enabled":true,"locations":["EEA"],"applicablePurposes":[{"id":1,"defaultValue":"DENIED"},{"id":2,"defaultValue":"DENIED"},{"id":3,"defaultValue":"DENIED"},{"id":4,"defaultValue":"DENIED"},{"id":5,"defaultValue":"DENIED"},{"id":6,"defaultValue":"DENIED"},{"id":7,"defaultValue":"DENIED"}]}},"premiumUiLabels":{},"premiumProperties":{"googleWhitelist":[1]},"coreUiLabels":{},"theme":{},"nonIabVendorsInfo":{}});})();