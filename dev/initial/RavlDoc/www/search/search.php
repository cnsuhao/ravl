<?php
/*

        © BY http://phpMySearch.web4.hm

        */
    
$Path = "";
include($Path."Src/class.FastTemplate.php");
include($Path."Src/class.SearchVisualizer.inc.php");
include($Path."Src/class.SearchEngine.inc.php");
include($Path."Src/class.Config.inc.php");
include($Path."Src/class.Const.inc.php");
include($Path."Src/class.AdminConfig.inc.php");
include($Path."Src/class.RevPolNotation.inc.php");
include($Path."Src/class.SpiderState.inc.php");
include($Path."Src/language.inc.php");
include($Path."Src/class.ExpressionPrepocessor.inc.php");
include($Path."Src/DataBaseSettings.inc.php");
include($Path."Src/class.Log.inc.php");

set_time_limit(300000);

$vis = new SearchVisualizer();

$vis->start("syntax");

echo $HtmlOutput;

?>