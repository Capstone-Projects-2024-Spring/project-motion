"use strict";(self.webpackChunk_1001fonts=self.webpackChunk_1001fonts||[]).push([[4832],{47244:(e,t,s)=>{var a=s(68188),r=s.n(a);const o=JSON.parse('{"base_url":"","routes":{"admin_get_activity_typeface":{"tokens":[["text",".json"],["variable","/","[^\\\\/]+","slug",true],["text","/admin/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"admin_get_activity_user":{"tokens":[["text",".json"],["variable","/","[^/\\\\.]++","username",true],["text","/admin/user"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"thread_edit":{"tokens":[["text","/edit"],["variable","/","\\\\d+","id",true],["text","/thread"],["variable","/","[^/]++","slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["GET","POST"],"schemes":[]},"get-editor-features":{"tokens":[["variable","/","[^/]++","type",true],["text","/editor-features"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_illustration_upload_status":{"tokens":[["variable","/","[^\\\\/]+","uuid",true],["text","/illustration/upload/status"]],"defaults":[],"requirements":{"uuid":"[^\\\\/]+"},"hosttokens":[],"methods":["GET"],"schemes":[]},"typeface_illustration_upload_files":{"tokens":[["text","/illustration/upload-files"],["variable","/","[^\\\\/]+","slug",true]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"typeface-search":{"tokens":[["text","/search.html"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"add_typeface_link":{"tokens":[["text","/link"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"delete_typeface_link":{"tokens":[["variable","/","-?\\\\d+","id",true],["text","/link"],["variable","/","[^/]++","typefaceSlug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+","id":"-?\\\\d+"},"hosttokens":[],"methods":["DELETE"],"schemes":[]},"typeface_favorites_count":{"tokens":[["text","/favorites/count"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_license_form":{"tokens":[["text","/form"],["variable","/","[^\\\\/]+","slug",true],["text","/licenses"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"get_typeface_activities":{"tokens":[["text","/activity/"],["variable","/","[^/]++","slug",true],["text","/typeface"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_links":{"tokens":[["text","/link"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_published_status":{"tokens":[["text","/is-published/"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_details":{"tokens":[["text","-font.html"],["variable","/","[^\\\\/]+","slug",true]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"homepage":{"tokens":[["text","/"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"index-tagged":{"tokens":[["text","-fonts.html"],["variable","/","[^\\\\/]+","tags",true]],"defaults":[],"requirements":{"tags":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_font_update_text":{"tokens":[["text","/text"],["variable","/","(?:[^\\\\/]+\\\\.){2}[^\\\\/]+","slug",true],["text","/typeface_font"]],"defaults":[],"requirements":{"slug":"([^\\\\/]+\\\\.){2}[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"typeface_charmaps":{"tokens":[["text","/chars"],["variable","/","(?:[^\\\\/]+\\\\.){2}[^\\\\/]+","slug",true],["text","/typeface_font"]],"defaults":[],"requirements":{"slug":"([^\\\\/]+\\\\.){2}[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_charmaps_update":{"tokens":[["text","/chars"],["variable","/","(?:[^\\\\/]+\\\\.){2}[^\\\\/]+","slug",true],["text","/typeface_font"]],"defaults":[],"requirements":{"slug":"([^\\\\/]+\\\\.){2}[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"author_statistics_typefaces":{"tokens":[["text","/statistics/typefaces"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"author_statistics_top10_downloaded":{"tokens":[["text","/statistics/top10/downloaded"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"author_statistics_top10_favorited":{"tokens":[["text","/statistics/top10/favorited"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"user_avatar":{"tokens":[["variable",".","[^/]++","_format",true],["text","/avatar"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":{"_format":"avif"},"requirements":[],"hosttokens":[["text","static.1001fonts.test"]],"methods":["GET"],"schemes":[]},"user_avatar_original":{"tokens":[["text","/avatar_original"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET"],"schemes":[]},"delete_user_avatar":{"tokens":[["text","/avatar"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["DELETE"],"schemes":[]},"update_user_avatar_service":{"tokens":[["text","/avatar-provider"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["POST"],"schemes":[]},"upload_user_avatar":{"tokens":[["text","/avatar"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["POST"],"schemes":[]},"email-check":{"tokens":[["variable","/","[^/]++","email",true],["text","/check/email"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"user_profile_json":{"tokens":[["text",".json"],["variable","/","[^/\\\\.]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"user_profile":{"tokens":[["text","/"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"username-check":{"tokens":[["variable","/","[^/]++","username",true],["text","/check/username"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"activities":{"tokens":[["text","/admin/activities"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"activities_by_day":{"tokens":[["variable","/","[^/]++","day",true],["text","/admin/activities"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"get_activity_actions":{"tokens":[["text","/admin/activity/action"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"boards_index":{"tokens":[["text","/forums/"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"board_index":{"tokens":[["text","/"],["variable","/","[^/]++","slug",true],["text","/forums"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"thread_show":{"tokens":[["text",".html"],["variable","/","\\\\d+","id",true],["variable","/","[^/]++","board_slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"thread_get":{"tokens":[["variable","/","\\\\d+","id",true],["variable","/","[^/]++","board_slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"update_thread":{"tokens":[["variable","/","\\\\d+","id",true],["variable","/","[^/]++","board_slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["PATCH","POST"],"schemes":[]},"post_reply":{"tokens":[["text","/reply"],["variable","/","\\\\d+","id",true],["variable","/","[^/]++","board_slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"delete_reply":{"tokens":[["variable","/","\\\\d+","id",true],["text","/forums/reply"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["DELETE"],"schemes":[]},"delete_thread":{"tokens":[["variable","/","\\\\d+","id",true],["variable","/","[^/]++","board_slug",true],["text","/forums"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["DELETE"],"schemes":[]},"update_reply":{"tokens":[["variable","/","\\\\d+","id",true],["text","/forums/reply"]],"defaults":[],"requirements":{"id":"\\\\d+"},"hosttokens":[],"methods":["PUT"],"schemes":[]},"typeface_comments":{"tokens":[["text","/comment"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_comment_delete":{"tokens":[["variable","/","\\\\d+","id",true],["text","/comment"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+","id":"\\\\d+"},"hosttokens":[],"methods":["DELETE"],"schemes":[]},"typeface_comment_add":{"tokens":[["text","/comment"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"typeface_change_license":{"tokens":[["text","/license"],["variable","/","[^\\\\/]+","slug",true],["text","/licenses"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"login":{"tokens":[["text","/sign-in.html"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","POST","HEAD"],"schemes":[]},"typeface_total_downloads":{"tokens":[["text","/statistics/downloads"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"filter_tags":{"tokens":[["variable","/","[^\\\\/]{2,}","filter",true],["text","/tag"]],"defaults":[],"requirements":{"filter":"[^\\\\/]{2,}"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_update_objects_order":{"tokens":[["variable","/","fonts|illustrations","objects",true],["text","/order"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+","objects":"fonts|illustrations"},"hosttokens":[],"methods":["POST"],"schemes":[]},"typeface_update_description":{"tokens":[["text","/description"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"change_typeface_author":{"tokens":[["text","/author"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"update_typeface_author":{"tokens":[["text","/author"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"filter_author":{"tokens":[["variable","/","[\\\\w\\\\W]{1,}","filter",true],["text","/typeface/author"]],"defaults":[],"requirements":{"filter":"[\\\\w\\\\W]{1,}"},"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"typeface_reupload":{"tokens":[["text","/reupload"],["variable","/","[^\\\\/]+","slug",true],["text","/typeface"]],"defaults":[],"requirements":{"slug":"[^\\\\/]+"},"hosttokens":[],"methods":["POST"],"schemes":[]},"user-tag-cloud":{"tokens":[["text","/tags.html"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"user_links":{"tokens":[["text","/link"],["variable","/","[^/]++","slug",true],["text","/users"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"bazinga_jstranslation_js":{"tokens":[["variable",".","js|json","_format",true],["variable","/","[\\\\w]+","domain",true],["text","/translations"]],"defaults":{"domain":"messages","_format":"js"},"requirements":{"_format":"js|json","domain":"[\\\\w]+"},"hosttokens":[],"methods":["GET"],"schemes":[]},"legacy_free-fonts-for-commercial-use":{"tokens":[["text","/free-fonts-for-commercial-use.html"]],"defaults":[],"requirements":[],"hosttokens":[],"methods":["GET","HEAD"],"schemes":[]},"local_avatar":{"tokens":[["variable",".","[^/]++","_format",true],["variable","_","[^/\\\\.]++","size",true],["variable","/","[^/_]++","slug",true],["text","/img/avatars"]],"defaults":[],"requirements":[],"hosttokens":[["text","static.1001fonts.test"]],"methods":["GET"],"schemes":[]},"local_avatar_old":{"tokens":[["variable",".","[^/]++","_format",true],["variable","/","[^/\\\\.]++","slug",true],["text","/img/avatars"]],"defaults":[],"requirements":[],"hosttokens":[["text","static.1001fonts.test"]],"methods":["GET"],"schemes":[]},"local_silhouette":{"tokens":[["text","/img/profile-silhouette.svg"]],"defaults":[],"requirements":[],"hosttokens":[["text","static.1001fonts.test"]],"methods":["GET"],"schemes":[]}},"prefix":"","host":"localhost","port":"","scheme":"http","locale":""}');r().setRoutingData(o)},65292:(e,t,s)=>{s(37588)}},e=>{var t=t=>e(e.s=t);e.O(0,[8432],(()=>(t(47244),t(65292))));e.O()}]);