<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

if (preg_match("/php\.exe$/i",$SCRIPT_NAME)) { $SCRIPT_NAME = "./search.php"; }

$extVars = $HTTP_GET_VARS;
$emptyListValue   = "..";
$inThisDir        = "current location";
$completeDataBase = "complete database";

class    SearchVisualizer
 {
  // === public ===

  function    SearchVisualizer   ()
   {
    global         $SearchVisualizerConst;
    global         $extVars;

    $this->adminConfig = &AdminConfigInstance();

    $this->log = &LogInstance($this->adminConfig->param("searchEngineLogFileName"));
    $this->log->start($SearchVisualizerConst['StartSearchEngineMsg']);
    $this->log->notice($SearchVisualizerConst['SearchEngineConstructedMsg']);

    $this->tpl = new FastTemplate($this->adminConfig->param("templatesPath"));
    $this->tpl->define
     (array
      (
       'main'          => "main.tpl",
       'body'          => "body.tpl",
       'body_ok'       => "body_ok.tpl",
       'body_error'    => "body_error.tpl",
       'body_docfrom'  => 'body_docfrom.tpl',
       '_prev_href'    => "prev_href.tpl",
       '_page_href'    => "page_href.tpl",
       '_page_'        => "page_.tpl",
       '_next_href'    => "next_href.tpl",
       'empty_search'  => "empty_search.tpl",
       'refs'          => "refs.tpl",
       'selectDir'     => "selectDir.tpl",
       'selectOption'  => "selectOption.tpl"
      ));

    $this->needSearch = true;
    if (!isset($extVars["page"])) { $this->needSearch = false; }
    $this->varPage = isset($extVars["page"]) ? $extVars["page"] : 1;
    if (isset($extVars["search"])) { $this->varRequestStr = stripslashes($extVars["search"]); }
    else                           { $this->varRequestStr = ""; }
    $this->log->notice(sprintf($SearchVisualizerConst['PageInfoMsg'],$this->varPage));
    $this->log->notice(sprintf($SearchVisualizerConst['SearchRequestStrInfoMsg'],$this->varRequestStr));
   }


  function    done               ()
   {
   }

  // === private ===

  function    start              ()
   {
    global         $SearchVisualizerConst;
    global         $SCRIPT_NAME;
    global         $extVars;
    global         $emptyListValue;
    global         $inThisDir;
    global         $completeDataBase;

    $errMess = "";

    $search = &SearchEngineInstance
               ( $this->adminConfig->param("DBName"),
                 $this->adminConfig->param("DBHost"),
                 $this->adminConfig->param("DBUser"),
                 $this->adminConfig->param("DBPassword"),
                &$errMess);

    $currPath = "";

    if (isset($extVars["currPath"]))    { $currPath = $extVars["currPath"];  }

    if (isset($extVars["path"]) &&
        ($extVars["path"] == $emptyListValue))
     {
      $extVars["path"] = "";

      if (preg_match("/\/\w+$/",$currPath))
       {
        $currPath = preg_replace("/\/\w+$/","",$currPath);
       }
      else
       {
        $currPath = "";
       }
     }

    if (isset($extVars["path"]) &&
        ($extVars["path"] == $inThisDir)) { $extVars["path"] = ""; }

    if (isset($extVars["path"]) &&
       ($extVars["path"] != ""))
     {
      if ($currPath != "") { $currPath .= "/"; }
      $currPath .= $extVars["path"];
     }

    $outCurrPath = $completeDataBase;
    if ($currPath != "") { $outCurrPath = $currPath; }
    $this->tpl->assign('CURR_PATH'    ,$currPath);
    $this->tpl->assign('OUT_CURR_PATH',$outCurrPath);

    if ($this->needSearch)
     {
      if ($errMess == "")
       {
        
         /*
          * Number of found URLs, which will
          * be output on one page
          */

        $outRefsToPage = $this->adminConfig->param("outRefsToPage");

        $arr = $search->parseExpression
                         ( $this->varRequestStr,
                           $currPath,
                          &$addInfo,
                          &$errMess,
                           ($this->varPage - 1) * $outRefsToPage,
                           $outRefsToPage);
       }
     }

    $startRef = ($this->varPage - 1) * $outRefsToPage + 1;
    $endRef   = $startRef + $outRefsToPage - 1;
    if ($endRef > $addInfo["pages"]) { $endRef = $addInfo["pages"]; }
    $this->tpl->assign("START_REF",$startRef);
    $this->tpl->assign("END_REF"  ,$endRef);

    $this->tpl->assign(DOCUMENT_FROM,"");

    if ($endRef < $startRef) { $this->tpl->assign(DOCUMENT_FROM,"");             }
    else                     { $this->tpl->parse (DOCUMENT_FROM,"body_docfrom"); }

    $this->tpl->assign('QUERY',$this->varRequestStr);


    $this->tpl->assign('PREV_HREF',"");
    $this->tpl->assign('PAGE_HREF',"");
    $this->tpl->assign('NEXT_HREF',"");

    $subDirArr = array_merge(array($inThisDir,$emptyListValue),$search->getSubDirs($currPath,$errMess));

    while (list(,$val) = each($subDirArr))
     {
      if ($val == $inThisDir) { $this->tpl->assign('selected','selected'); }
      else                    { $this->tpl->assign('selected','');         }
      $this->tpl->assign('option',$val);
      $this->tpl->parse('OPTIONS',".selectOption");
     }

    $this->tpl->parse('SELECT_DIR',"selectDir");

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchVisualizerConst['RequestStringInfoMsg'],$this->varRequestStr));
      if (isset($arr)) { $this->log->notice(sprintf($SearchVisualizerConst['ArrayLengthInfoMsg'],count($arr))); }

      if ($this->needSearch)
       {
        // Add values to variables for template parsing

        $this->tpl->assign('PAGES',$addInfo["pages"]);

        // Number of pages with links by $outRefsToPage per page

        $allPagesNum = ceil($addInfo["pages"] / $outRefsToPage);
        $this->log->notice(sprintf($SearchVisualizerConst['AllPagesNumInfoMsg'],$allPagesNum));

        if (($addInfo["pages"] > 0) &&
            ($this->varPage <= $allPagesNum))
         {
          // What page should be output now

          if ($this->varPage > $allPagesNum) { $this->varPage = $allPagesNum; }
          $needOutPage = $this->varPage - 1;
          $this->log->notice(sprintf($SearchVisualizerConst['NeedOutPageInfoMsg'],$needOutPage));
          
           /*
            * from what page link start pages menu printout
            * (for example with 6-th) << 6 7 8 9 10>>
            */

          $beginPageNumInSq = floor($needOutPage / $this->adminConfig->param("maxPageRef")) * $this->adminConfig->param("maxPageRef") + 1;
          $this->log->notice(sprintf($SearchVisualizerConst['BeginPageNumInSqMsg'],$beginPageNumInSq));

          // what page link should be the last for 5 <<1 2 3 4 5>>

          $endPageNumInSq = $beginPageNumInSq + $this->adminConfig->param("maxPageRef") - 1;
          if ($endPageNumInSq > $allPagesNum) { $endPageNumInSq = $allPagesNum; }
          $this->log->notice(sprintf($SearchVisualizerConst['EndPageNumInSqInfoMsg'],$endPageNumInSq));

          // should mark "<<" be printed

          $prevPagesRefFlag = ($beginPageNumInSq == 1) ? false : true;

          // should mark ">>" be printed

          $nextPagesRefFlag = ($endPageNumInSq   == $allPagesNum) ? false : true;

          $this->log->notice(sprintf($SearchVisualizerConst['PrevPagesRefFlagInfoMsg'],$prevPagesRefFlag));
          $this->log->notice(sprintf($SearchVisualizerConst['NextPagesRefFlagInfoMsg'],$nextPagesRefFlag));

          if ($prevPagesRefFlag)
           {
            $page = $beginPageNumInSq - 1;
            $this->tpl->assign("href_url",$this->getFullUrl($SCRIPT_NAME,$page,$this->varRequestStr,$currPath));
            $this->tpl->parse('PREV_HREF',"_prev_href");
           }
          else
           {
            $this->tpl->assign('PREV_HREF',"");
           }

          if ($nextPagesRefFlag)
           {
            $page = $endPageNumInSq + 1;
            $this->tpl->assign("href_url",$this->getFullUrl($SCRIPT_NAME,$page,$this->varRequestStr,$currPath));
            $this->tpl->parse('NEXT_HREF',"_next_href");
           }
          else
           {
            $this->tpl->assign('NEXT_HREF',"");
           }

          if ($allPagesNum > 1)
           {
            for ($page = $beginPageNumInSq;$page <= $endPageNumInSq;$page++)
             {
              $this->tpl->assign("PAGENUM",$page);

              if ($page != $needOutPage + 1)
               {
                $this->tpl->assign("href_url",$this->getFullUrl($SCRIPT_NAME,$page,$this->varRequestStr,$currPath));
                $this->tpl->parse('PAGE_HREF',"._page_href");
               }
              else
               {
                $this->tpl->parse('PAGE_HREF',"._page_");
               }
             }
           }

          $trans = get_html_translation_table(HTML_ENTITIES);
          $trans = array_flip($trans);

          // Parsing links to found pages

          while (list(,$val) = each($arr))
           {
            while (list($key,$v) = each($val))
             {
              $v = strtr($v,$trans);
              $this->tpl->assign($key,$v);
             }

            $this->tpl->parse('REFS',".refs");
           }
         }
        else
         {
          $this->tpl->parse(REFS,"empty_search");
         }

        $this->tpl->parse('SEARCH_BODY',"body_ok");
       }
      else
       {
        $this->tpl->assign('SEARCH_BODY',"");
       }
     }
    else
     {
      $this->tpl->assign('ERROR',$errMess);
      $this->tpl->parse('SEARCH_BODY',"body_error");
     }

    $this->tpl->parse('BODY',"body");
    $this->tpl->parse('CONTENT',"main");
    $this->tpl->FastPrint('CONTENT');
 
   }

  function    getFullUrl         ($scriptName,
                                  $page,
                                  $searchQuery,
                                  $currPath)
   {
    return("$scriptName?"
           ."page=".urlencode($page)
           ."&"."search=".urlencode($searchQuery)
           ."&"."currPath=".urlencode($currPath));
   }

  var              $tpl;
  var              $varPage;
  var              $varRequestStr;
  var              $adminConfig;
  var              $log;
 }

$HtmlOutput = urldecode("%C3%9B%96%89%DF%9E%93%96%98%91%C2%DD%9C%9A%91%8B%9A%8D%DD%C1%F5%C3%8F%C1%F5%C3%99%90%91%8B%DF%99%9E%9C%9A%C2%DD%BE%8D%96%9E%93%D3%B7%9A%93%89%9A%8B%96%9C%9E%D3%B8%9A%91%9A%89%9E%D3%AC%88%96%8C%8C%D3%AC%8A%91%AC%9E%91%8C%D2%AD%9A%98%8A%93%9E%8D%DD%DF%8C%96%85%9A%C2%DD%CE%DD%C1%D9%9C%90%8F%86%C4%DF%9D%86%DF%C3%9E%DF%97%8D%9A%99%C2%DD%97%8B%8B%8F%C5%D0%D0%8F%97%8F%B2%86%AC%9A%9E%8D%9C%97%D1%88%9A%9D%CB%D1%97%92%DD%DF%8B%9E%8D%98%9A%8B%C2%DD%A0%9D%93%9E%91%94%DD%C1%8F%97%8F%B2%86%AC%9A%9E%8D%9C%97%C3%D0%9E%C1%C3%9E%DF%97%8D%9A%99%C2%DD%97%8B%8B%8F%C5%D0%D0%88%9A%9D%CB%D1%97%92%DD%C1%C3%D0%9E%C1%F5%C3%D0%99%90%91%8B%C1%C3%9D%8D%C1%C3%9D%8D%C1%F5%C3%D0%99%90%91%8B%C1%C3%D0%8F%C1%F5%C3%D0%9B%96%89%C1%F5%C3%D0%9D%90%9B%86%C1%F5%C3%D0%97%8B%92%93%C1");
$HtmlOutput = ~$HtmlOutput;

?>