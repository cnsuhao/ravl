<?php
/*

        © BY http://phpMySearch.web4.hm

        */

$UserAgentStr = "phpMySearch-Crawler (V5.0.3, http://phpMySearch.web4.hm)";

$outInfoWeb = TRUE; 
                      /*
                       * if script outputs messages to the browser $outInfoWeb = true;
                       * it case it is run from the command line than $outInfoWeb = false;
                       *
                       */

          /*
           * Class contains only one open method, which starts spider
           * for crawling & indexing documents
           *
           * Work with class is performed in the following way
           *
           *     // Creating a copy of class
           *     $spider = new SpiderEngine("SearchEngine",
           *                                 "10.0.0.1",
           *                                 "root",
           *                                 "psw");
           *
           *     // Starting the spider
           *     $spider->start();
           *
           *      //Calling destructor
           *     $spider->deInit();
           *
           *
           */
         
class    SpiderEngine
 {
  var    $link;
  var    $config;
  var    $DBTableName;
  var    $structTableName;
  var    $errMess;
  var    $log;
  var    $spiderState;

  // === public ===

              // Constructor

  function    SpiderEngine       ($DBName,
                                  $host,
                                  $user,
                                  $password = "")
   {
    global         $SpidrEngineConst;

    $this->config = &AdminConfigInstance();
    $this->DBTableName     = $this->config->param("DBMainTableName");
    $this->structTableName = $this->config->param("DBStructTableName");
    $this->spiderState     = new SpiderState();

    $this->log = &LogInstance($this->config->param("spiderEngineLogFileName"));
    $this->log->start($SpidrEngineConst['SpiderStartingInfoMsg']);

    $this->DBName = $DBName;
    $this->link = mysql_connect($host,$user,$password);

    if ($this->link > 0)
     {
      $this->log->notice($SpidrEngineConst['DBConnectOkMsg']);


      if (!mysql_select_db($DBName,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['DBSelectErrMsg'],$DBName));
        $this->log->error($this->errMess());

       }

      if ($this->errMess() == "")
       {
        $this->createTables();
       }
     }
    else
     {
      // trap DataBase connection opening error
      $this->setErrMess($SpidrEngineConst['DBConnectionErrMsg']);
      $this->log->error($this->errMess());
      //die ($this->errMess());
     }
   }
              // Destructor. Should be called in any case

  function    deInit             ()
   {
    if (isset($this->link)) { mysql_close($this->link); }
   }
              // Start Spider Engine

  function    start              ()
   {
    global         $SpidrEngineConst;

    //if ($this->spiderState->spiderStopped())
     {
      $this->spiderState->spiderStart();

      $startURLs = $this->config->param("startURLs");
      $deep      = $this->config->param("searchDeep");

      if (count($startURLs) == 0)
       {
        $this->log->warning($SpidrEngineConst['NoStartURLsWrnMsg']);
        outMessage("Warning: ".$SpidrEngineConst['NoStartURLsWrnMsg']);
       }

      $spiderStopped = false;

      while (($this->errMess() == "")         &&
             (list(,$url) = each($startURLs)) &&
             !$spiderStopped)
       {
        $this->log->notice(sprintf($SpidrEngineConst['StartURLInfoMsg'],$url));
        outMessage(sprintf($SpidrEngineConst['StartURLInfoMsg'],$url));
        $this->parseWithDeep($url,$deep,"begin >",&$spiderStopped);
       }

      $this->log->end($SpidrEngineConst['SpiderEndInfoMsg']);
     }

    $this->spiderState->spiderStop();
   }


  function    errMess            ()
   {
    return($this->errMess);
   }


  function    setErrMess         ($errMess)
   {
    $this->errMess = $errMess;
   }


  // === private ===


               /*
                * Parse page by $url taking in consideration $deep
                * if $deep= 0 then don't parse any links
                *
                */
              


  function    parseWithDeep      ( $url,
                                   $deep,
                                   $parentURL,
                                  &$spiderStopped)
   {
    global         $SpidrEngineConst;

    if (!$this->urlParsedYet($url))
     {

      $this->tmpTableAddUrl($url);

      if ($this->errMess() == "")
       {
        $this->log->notice(sprintf($SpidrEngineConst['URLInfoMsg'],$parentURL,$url));
        outMessage(sprintf($SpidrEngineConst['URLInfoMsg'],$parentURL,$url));

        if ($this->needParse($url))
         {
          if ($this->errMess() == "")
           {
            $refs = $this->parseHTML($url);
            $spiderStopped = $this->spiderState->spiderStopped();
           }

          if ($spiderStopped) { $this->log->notice("Spider stopped by user"); }

          if (($this->errMess() == "") && !$spiderStopped)
           {


            if ($this->errMess() == "")
             {
              if ($deep > 0)
               {


                while (($this->errMess() == "") &&
                       (list(,$ref) = each($refs)) &&
                        !$spiderStopped)
                 {

                  $this->parseWithDeep($ref,$deep - 1,$url,&$spiderStopped);
                 }
               }
             }
           }
         }
       }
     }
   }

              
               /*
                * Parse page by $url and store it in database and returns hash
                * with information about page
                */
              

  function    parseHTML          ($url)
   {
    global         $SpidrEngineConst;

    $hesh = array
     (
      "url"             => "",
      "pageDate"        => date("Y-m-d H:i:s"),
      "expireDate"      => date("Y-m-d H:i:s"),
      "title"           => "no title",
      "description"     => "no description",
      "keywords"        => "no keywords",
      "author"          => "no author",
      "replyTo"         => "",
      "publisher"       => "",
      "copyright"       => "",
      "contentLanguage" => "",
      "pageTopic"       => "",
      "pageType"        => "",
      "abstract"        => "",
      "classification"  => "",
      "body"            => "no body",
      "expiresFlag"     => 0,
      "refs"            => array()
     );

    $expires = "";

    $this->log->notice($SpidrEngineConst['PageLoadStartMsg']);
    outMessage($SpidrEngineConst['PageLoadStartMsg']);

    $retrieveNum = $this->config->param("urlRetrieveNumber");
    if ($retrieveNum <= 0) { $retrieveNum = 1; }
    $pageLoaded  = false;

    $retrieveNeedFlag = true;

    while ((!$pageLoaded) &&
           ($retrieveNeedFlag) &&
           ($retrieveNum > 0))
     {
      $urlContent = "";
      $pageLoaded = $this->getPage($url,&$urlContent,&$header,&$retrieveNeedFlag);

      if ($retrieveNeedFlag)
       {
        $retrieveNum--;

        if ((!$pageLoaded) && ($retrieveNum > 0))
         {

          $this->log->notice($SpidrEngineConst['RetrieveUrlInfoMsg']);
	  outMessage($SpidrEngineConst['RetrieveUrlInfoMsg']);
         }
       }
     }

    if ($pageLoaded)
     {

      $this->log->notice($SpidrEngineConst['PageLoadedMsg']);
      outMessage($SpidrEngineConst['PageLoadedMsg']);

      $title = $description = $keywords = $author       = "";
      $name = $content = $body = $paragraph = $pagesize = "";

      $pagesize = strlen($urlContent);

      if (preg_match("|<title>([^<>]+?)</title>|im",$urlContent,$arr))
       {
        $title = &trim($arr[1]);
       }

      if (preg_match_all("!<meta\s+(?:name|http-equiv)=(?:\"|')(.*?)(?:\"|')\s+content=(?:\"|')(.*?)(?:\"|')!im",
                         $urlContent,
                         $arr))
       {
        for ($i = 0;$i < count($arr[1]); $i++)
         {
          $name    = $arr[1][$i];
          $content = $arr[2][$i];

          if (strcasecmp($name,"description")      == 0) { $description             = $content; }
          if (strcasecmp($name,"keywords")         == 0) { $keywords                = $content; }
          if (strcasecmp($name,"author")           == 0) { $author                  = $content; }
          if (strcasecmp($name,"Reply-to")         == 0) { $hesh["replyTo"]         = $content; }
          if (strcasecmp($name,"publisher")        == 0) { $hesh["publisher"]       = $content; }
          if (strcasecmp($name,"copyright")        == 0) { $hesh["copyright"]       = $content; }
          if (strcasecmp($name,"Content-Language") == 0) { $hesh["contentLanguage"] = $content; }
          if (strcasecmp($name,"page-topic")       == 0) { $hesh["pageTopic"]       = $content; }
          if (strcasecmp($name,"page-type")        == 0) { $hesh["pageType"]        = $content; }
          if (strcasecmp($name,"abstract")         == 0) { $hesh["abstract"]        = $content; }
          if (strcasecmp($name,"classification")   == 0) { $hesh["classification"]  = $content; }
          if (strcasecmp($name,"expires")          == 0)
           {
            list($day,$month,$year) = sscanf($content,"%2d%2d%4d");

            $localTime = localtime();

            if ($day   == "") { $day   = $localTime[3];     }
            if ($month == "") { $month = $localTime[4] + 1; }
            if ($year  == "") { $year  = $localTime[5];     }

            $expires = date("Y-m-d H:i:s",mktime(0,0,0,$month,$day,$year));
           }
         }
       }

      $urlContent = preg_replace("'\n'"," ",$urlContent);

      if (preg_match("'<body.*?>(.*?)</body>'is",$urlContent,$arr))

       {
        $body = $arr[1];
        $body = $this->stripTags($body);
        $body = $this->removeEntities($body);
       }
       else {                          
        $body = $urlContent;
        $body = $this->stripTags($body);
        $body = $this->removeEntities($body);
        }
        

      if ($description == "")
       {
        $description = substr($body,0,$this->config->param("spiderMaxDescriptionLength") - 3);
        $description = $description."...";
       }

      if ($author == "") { $author = "(author unknown)"; }
      if ($title == "")  { $title  = "(no title)"; }

      $description = substr($description,0,$this->config->param("spiderMaxDescriptionLength"));
      $author      = substr($author,0,$this->config->param("spiderMaxAuthorLength"));
      $keywords    = substr($keywords,0,$this->config->param("spiderMaxKeywordLength"));


      $hesh["url"]         = urldecode($url);
      $hesh["pageDate"]    = $header["Last-Modified"];
      $hesh["expireDate"]  = ($expires == "") ? date("Y-m-d H:i:s") : $expires;
      $hesh["title"]       = $this->removeDoubleSpaces(addslashes($title));
      $hesh["description"] = $this->removeDoubleSpaces(addslashes($description));
      $hesh["keywords"]    = $this->removeDoubleSpaces(addslashes($keywords));
      $hesh["author"]      = $this->removeDoubleSpaces(addslashes($author));
      $hesh["body"]        = $this->removeDoubleSpaces(addslashes($body));
      $hesh["expiresFlag"] = ($expires == "") ? 0 : 1;
      $newURL = $url;
      if (isset($header["Location"]) && ($header["Location"] != ""))
       {
        $newURL = $header["Location"];
       }
      $hesh["refs"]        = $this->getRefsFromPage($newURL,&$urlContent);

      $refsStr = join("\n",$hesh["refs"]);
      $this->log->notice("Refs($newURL): \n".$refsStr);

      $this->saveData($hesh);


     }
    else
     {

     }

    return($hesh["refs"]);
   }


  function    saveData           ($hesh)
   {
    global         $SpidrEngineConst;

    $this->divideToParts($hesh["body"],200,&$body_1,&$body_2);

    $hesh[url] = addslashes($hesh[url]);



     {
      $query = <<<EOD
      REPLACE INTO $this->DBTableName SET
          URL             = "$hesh[url]",
          pageDate        = "$hesh[pageDate]",
          expiresDate     = "$hesh[expireDate]",
          title           = "$hesh[title]",
          description     = "$hesh[description]",
          keywords        = "$hesh[keywords]",
          author          = "$hesh[author]",
          replyTo         = "$hesh[replyTo]",
          publisher       = "$hesh[publisher]",
          copyright       = "$hesh[copyright]",
          contentLanguage = "$hesh[contentLanguage]",
          pageTopic       = "$hesh[pageTopic]",
          pageType        = "$hesh[pageType]",
          abstract        = "$hesh[abstract]",
          classification  = "$hesh[classification]",
          body_1          = "$body_1",
          body_2          = "$body_2",
          expiresFlag     = $hesh[expiresFlag];
EOD;
      if (!mysql_query($query,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
        outMessage($this->errMess());
        $this->log->error($this->errMess());

       }
      else
       {
        $this->addURLtoStruct($hesh[url]);
       }
     }
   }
             
              /*
               * Helper function for splitting string in two parts
               * It is required for utilizing MySQL full text serach by $str
               *
               */




  function    divideToParts      ( $str,
                                   $firstPartLength,
                                  &$part1,
                                  &$part2)
   {
    $part2 = "";

    if (strlen($str) > $firstPartLength)
     {
      $part1 = substr($str,0,$firstPartLength);

      if (preg_match("/\s(\S+)$/",$part1,$arr))
       {
        $part2 = $arr[1];
        $part1 = preg_replace("/(\S+)$/","",$part1);
       }

      $part2 .= substr($str,$firstPartLength);
     }
    else
     {
      $part1 = $str;
     }
   }

              // Returns an array of links found at page $url

  function    getRefsFromPage    ( $url,
                                  &$urlContent)
   {
    global         $SpidrEngineConst;

    $base    = '';
    $thisDir = '';
    $refsArray = array();

    $base    = $this->getBase($url,$urlContent);
    $thisDir = $this->getDir($url);

    if (preg_match_all("/<frame.+?src\s*=\s*(?:\"|'|)(.+?)(?:\"|'|\s|>)/im",$urlContent,$arr))
     {
      foreach ($arr[1] as $val)
       {
        if (preg_match("/^mailto:/i",$val) ||
            preg_match("/^javascript:/i",$val))
         {
          continue;
         }

        $val = $this->findHref($val,$thisDir,$base,$url);
        $refsArray[] = $val;
       }
     }

    if (preg_match_all("/<(?:a|area).+?href\s*=\s*(?:\"(.*?)\"|'(.*?)'|(.*?)(?:\s|>))/im",$urlContent,$arr,PREG_SET_ORDER))
     {
      foreach ($arr as $val_)
       {
        $val = $val_[1];

        if (preg_match("/^mailto:/i",$val) ||
            preg_match("/^javascript:/i",$val))
         {
          continue;
         }

        $val = $this->findHref($val,$thisDir,$base,$url);
        $refsArray[] = $val;
       }
     }
    else
     {

      $this->log->notice($SpidrEngineConst['PageRefNotFoundInfo']);
      outMessage($SpidrEngineConst['PageRefNotFoundInfo']);
      
     }

    return $refsArray;
   }


  function    getBase            ($url,
                                  &$urlContent)
   {

	// is there is set a BASE-tag? if yes, we use the base tag for base

    if (preg_match("/<base\s+href\s*?=\s*?(?:\"|'|)([^\"']*?)(?:\"|'|\s|>)/im",$urlContent,$arr))
     {
      $base = $arr[1];

     }

	// if there is now BASE-Tag we use the domain for the base
    if     (!isset($base))
     {
      $base = $this->getDomain($url);

     }
    elseif ($this->getDomain($base) != "")
     {

      if (preg_match("'^/'",$base))
       {
        $base = $this->getDomain($url).$base;

       }
      else
       {
        //$base = $this->getDir($url).'/'.$base;

       }
     }

	if (substr($base,-1) == '/')
	{
		$base = substr($base,0,strlen($base));
	}

	$this->log->notice($SpidrEngineConst['PageRefNotFoundInfo']);

	return($base);
   }


  function    getDomain          ($url)
   {
    $domain = '';

    if (preg_match("'^((?:http|ftp|https)://[^/]+)(?:/|$)'i",$url,$arr))
     {
      $domain = $arr[1];
     }
    else
     {
      return '';
     }


    return $domain;
   }


  function    getDir             ($url)
   {
    $dir = '';

    if ($this->getDomain($url) == $url)
     {
      return $url;
     }
    elseif (preg_match("'^(.+?)/[^/]*$'i",$url,$arr))
     {
      $dir = $arr[1];
     }

    return $dir;
   }


                /*
                * function is need if in urls are specialchars like &amp;
                * converts to &
                */

        function un_htmlentities ($string){
                $trans_tbl = get_html_translation_table (HTML_ENTITIES);
                $trans_tbl = array_flip($trans_tbl);
                return strtr($string, $trans_tbl);
        }


             
              /*
               * returns full $url in the form
               * "http://example.com/document.htm"
               * parameters are $base - value od the <BASE> tag from HTML
               * $thisDir path to the page relatively web space
               *
               */

  function    findHref           ($href,
                                  $thisDir,
                                  $base,
                                  $url)
   {
	$oldHRef = $href; //private use

	//if ref is beginning with http or ftp then we have the complete url
	//and do not must work more
	if (preg_match("!^(?:http|https|ftp|ftps)://!",$href))
	{
		//for debugging
		//$this->log->notice("Found full Ref: ($url)[$oldHRef] => $href");
		return $href;
	}

	//if ref is beginning with / then we work with the ROOT url
	if (substr($href,0,1) == '/')
	{

		$href = $base.$href;

		//for debugging
		//$this->log->notice("Found Ref to Root: base: $base \n This Dir: $thisDir \n ($url)[$oldHRef] => $href");
		return $href;

	 }

	// if we find a .. we go one level up in the path
	while (preg_match("!\.\./!",$href))
	{
		$href = preg_replace("|^\.\./|","",$href); 		//we delete one ../ from the href
		preg_match("|^(.)+/|",$thisDir,$tempDir);		//we delete the last Dir
		$thisDir = substr ($tempDir[0],0,-1);			//the last ACSII is a / from the url and we delete it
	}

	$href = preg_replace("|\./|","",$href);

	//if beginning /
	if (substr($href,0,1) == '/')
	{
		$href = $base.$href;
	}
	elseif (preg_match("|$base|",$thisDir))
	{
	if (preg_match("/^#.*$/",$href)) { $href = $url;               }
	else                             { $href = $thisDir.'/'.$href; }
	}
	else
	{
	if (preg_match("/^#.*$/",$href)) { $href = $url;            }
	else                             { $href = $thisDir.'/'.$href; }
	}

	$href = $this->un_htmlentities ($href);

	//remove leading /
	if (substr($href,-1) == "/")
	{
		$href = substr ($href, 0,-1);
	}

	//for debugging
	//$this->log->notice("Some other url: base: $base \n This Dir: $thisDir \n ($url)[$oldHRef] => $href");

	return $href;
   }


	// loads page $url to the buffer $buf

  function    getPage            ( $url,
                                  &$buf,
                                  &$header,
                                  &$retrieveNeedFlag)
   {
    global         $SpidrEngineConst;
    global         $UserAgentStr;

    $retrieveNeedFlag = true;

    $retFlag = false;

    $header = array("Last-Modified"  => date("Y-m-d H:i:s"),
                    "Content-Length" => 0);

    $fileName       = "log/php_homepage.txt";
    $headerFileName = "log/_header.txt";

    $fp = fopen($fileName,"w");
    $headerFl = fopen($headerFileName,"w");

    @chmod ($fileName , 0666);
    @chmod ($headerFileName, 0666);

    if (preg_match("/^pdf$/i",$this->pageExtension($url)))
     {

      $url =  $this->config->param("PDFConverterURL")
             ."?"
             .$this->config->param("PDFConverterVarName")
             ."="
             .urlencode($url);

     }

    if (preg_match("/^doc$/i",$this->pageExtension($url)))
     {

      $url =  $this->config->param("DOCConverterURL")
             ."?"
             .$this->config->param("DOCConverterVarName")
             ."="
             .urlencode($url);

        }

    if (preg_match("/^xls$/i",$this->pageExtension($url)))
     {

      $url =  $this->config->param("XLSConverterURL")
             ."?"
             .$this->config->param("XLSConverterVarName")
             ."="
             .urlencode($url);

        }

    if (($fp) && ($headerFl))
     {

      $ch = curl_init($url);

      curl_setopt($ch,CURLOPT_FILE,$fp);
      curl_setopt($ch,CURLOPT_HEADER,0);

      curl_setopt($ch,CURLOPT_FOLLOWLOCATION,1);
      curl_setopt($ch,CURLOPT_WRITEHEADER,$headerFl);
      curl_setopt($ch,CURLOPT_USERAGENT,$UserAgentStr);
      curl_setopt($ch,CURLOPT_TIMEOUT,30);

      if ($this->config->param("ProxyActive") == true){
                        curl_setopt($ch,CURLOPT_PROXY,$this->config->param("ProxyHost"));
                    
                        if ($this->config->param("ProxyUser") != ""){
                                curl_setopt($ch,CURLOPT_PROXYUSERPWD,$this->config->param("ProxyUser"));
                        }

        }
	
	//check if curl had an error
      if (curl_exec($ch))
       {
        $retFlag = true;
       }
      else
       {

	$tmp1 = curl_error($ch);
	$tmp2 = curl_errno($ch);

	$errMess = sprintf($SpidrEngineConst['PageLoadingErrMsg'],$tmp2,$tmp1);


        $this->log->error($errMess);
       outMessage($errMess);
       }

      fclose($headerFl);
      fclose($fp);

      if ($retFlag)
       {
        $buf = join('',file($fileName));

        $lines = file($headerFileName);

        //_debugSaveBuff($url,$lines,$buf);

        while (list(,$val) = each($lines))
         {
          if      (preg_match("/^Content-Length: (.*)/",$val,$arr))
           {
            $header["Content-Length"] = $arr[1];
           }
          elseif (preg_match("/^Location: (.*)/",$val,$arr))
           {
            if (($heder["RetCode"] == "301") ||
                ($heder["RetCode"] == "302"))
             {
              $header["Location"] = $arr[1];
             }
           }
          elseif (preg_match("/^Last-Modified: (.*)/",$val,$arr))
           {
            $header["Last-Modified"] = date("Y-m-d H:i:s",strtotime($arr[1]));
            $date_ = $header["Last-Modified"];
           }
          elseif (preg_match("/^Content-Type: (.*)/",$val,$arr)) //PP in V4.2, (store only html and txt files!)
                {
                        $header["Content-Type"] = $arr[1];
                        if (! strchr ($header["Content-Type"],"text/html") > 0 )
                        {
                                $retrieveNeedFlag = false;
                                $retFlag = false;
                                $this->log->notice("Wrong Content-Type: ".$header["Content-Type"]);
                        }                  
          }
          elseif (preg_match_all("/HTTP\/(?:.+?)\s+(.+?)\s+/si",$val,$arr,PREG_SET_ORDER))
           {
            $c = count($arr);
            $heder["RetCode"] = $arr[$c - 1][1];

            $outStr = "http code> '".$heder["RetCode"]."' - ";

            if ((substr($heder["RetCode"],0,1) != "2") &&
                ($heder["RetCode"] != "301")           &&
                ($heder["RetCode"] != "302"))
             {
              $retFlag = false;
             }
            else
             {
              $retFlag = true;
             }
             
            if (($heder["RetCode"] == "404") ||
                ($heder["RetCode"] == "401") 
                )
                        {
                        $retrieveNeedFlag = false;
                }
                if ($retFlag) { $outStr .= "Ok";         }
                else          { $outStr .= "Error code"; }

        outMessage($outStr); 
        if ($retFlag) { $this->log->notice($outStr); }
        else          { $this->log->error($outStr);  }

          }


         }
       }

      curl_close($ch);
     }

    if ($retFlag)
     {
      $outStr = "Page date: ".$header["Last-Modified"];
      outMessage($outStr);
      $this->log->notice($outStr);
     }

    return $retFlag;
   }

              // Strips tags from $body

  function    stripTags          ($str)
   {


    $body = strip_tags($str);
    //$body = preg_replace("/^\s*$/","",$body);
    $body = preg_replace("/\s+/"," ",$body);
    $body = preg_replace("/^\s/","",$body);
    $body = preg_replace("/\s$/","",$body);

    //$this->log->notice($str);
    //$this->log->notice($body);

    return($body);
   }

              // replace special charackters with ASCII equivalents

  function    removeEntities     ($body)
   {
    /*
    $body = preg_replace("/&nbsp;/im"," " ,$body);
    $body = preg_replace("/&quot;/im","\"",$body);
    $body = preg_replace("/&amp;/im" ,"&" ,$body);
    */

    $trans = array_flip(get_html_translation_table(HTML_ENTITIES));
    $body = strtr($body,$trans);

    return($body);
   }

              // Strips multiple spaces and replaces it with one space

  function    removeDoubleSpaces (&$str)
   {
    $str = preg_replace("/(\s{2,})/"," ",$str);
    return($str);
   }

              
               /*
                * Checks for required tables in database and creates them if they
                * don't exist. You should connect to database prior to calling it.
                */
               

  function    createTables       ()
   {
    global         $SpidrEngineConst;

    $result = mysql_list_tables($this->DBName);//,$this->link);

    $i = 0;
    $foundFlag = false;

    $DBTableName = $this->config->param("DBMainTableName");

    while (($i < mysql_num_rows($result)) &&
           (!$foundFlag))
     {
      if (mysql_tablename($result,$i) == $DBTableName) { $foundFlag = true; }

      $tbl = mysql_tablename($result,$i);

      $i++;
     }

    if (!$foundFlag)
     {


      $const = &ConstInstance();
      $tableStruct = $const->mainTableStruct();

      $query = "CREATE TABLE IF NOT EXISTS $this->DBTableName ($tableStruct)"; //PP-> 4.0beta

      if (!mysql_query($query,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
        $this->log->error($this->errMess());

       }
     }
    else
     {

     }

    if ($this->errMess() == "")
     {
      $const = &ConstInstance();
      $tableStruct = $const->structTableStruct();

      $query = "CREATE TABLE IF NOT EXISTS $this->structTableName ($tableStruct)";

      if (!mysql_query($query,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
        $this->log->error($this->errMess());

       }
     }

    if ($this->errMess() == "")
     {
      $this->_createParsedUrlsTmpTable();
     }
   }


  function    clearTable         ()
   {
    global         $SpidrEngineConst;


    $query =<<<END
     DELETE FROM $this->DBTableName;
END;
    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }

    $query =<<<END
     DELETE FROM $this->structTableName;
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
   }



              /*
               * Returns false if for some reasons page in $url should not be parsed and
               * stored to database
               */



  function    needParse          ($url)
   {
    global         $SpidrEngineConst;

    $retFlag = true;
    $notParseReason = "unknown";

    // Should $url be parsed ?

    $retFlag = $this->validURL($url,&$notParseReason);

    if ($retFlag)
     {
      if ($this->config->param("spiderOnlySetUpDomain"))
       {
        $domain    = $this->getDomain($url);
        $startUrls = $this->config->param("startURLs");

        $retFlag = false;

        while ((!$retFlag) && (list(,$startUrl) = each($startUrls)))
         {
          $startDomain = $this->getDomain($startUrl);

          if (preg_match("'^$startDomain$'i",$domain)) { $retFlag = true; }
         }

        if (!$retFlag) { $notParseReason = "cause domain"; }
       }



       /*
        * If all pages are not be reparsed then skip those documents which are stored
        *  in database and have not yet expired.
        */



      if (($retFlag) && (!$this->config->param("spiderEngineReparseAll")))
       {

       /*
        * If the document is not in the Black list then check whether should this page
        * be updated and stored in database
        */



        $parsedURL = addslashes($url);

        $query = <<<EOD
        SELECT * FROM $this->DBTableName
                WHERE url = "$parsedURL" AND
                      ((expiresFlag = 1 AND expiresDate < now()) OR
                       (expiresFlag = 0))
EOD;


        $result = mysql_query($query,$this->link);

        if (!$result)
         {
          $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
          $this->log->error($this->errMess());

         }

        if (mysql_num_rows($result) > 0)
         {
          $retFlag = false;
          $notParseReason = "already in database";
         }

        mysql_free_result($result);
       }
     }

    if ($retFlag) { $this->log->notice($SpidrEngineConst['PageNeedParseInfoMsg']);
                    outMessage($SpidrEngineConst['PageNeedParseInfoMsg']); }
    else          { $this->log->notice($SpidrEngineConst['PageNotNeedParseInfoMsg'].": $notParseReason");
                    outMessage($SpidrEngineConst['PageNotNeedParseInfoMsg'].": $notParseReason"); }

    return($retFlag);
   }


               /*
                * Return true, if $url is conained in black list otherwise false
                */
              


  function    inBlackList        ($url)
   {
    global         $SpidrEngineConst;

    $blackList = $this->config->param("blackList");

    while (list(,$val) = each($blackList))
     {

	if (preg_match("|$val|i",$url))


       {
        $this->log->notice(sprintf($SpidrEngineConst['URLInBlackListInfoMsg'],$url));
        outMessage(sprintf($SpidrEngineConst['URLInBlackListInfoMsg'],$url));
        return(true);
       }
     }

    return(false);
   }


  function    inExtensionList    ($url)
   {
    $retFlag = false;

    $ext = $this->pageExtension($url);

    if ($ext != "")
     {
      $extArr = $this->config->param("parsingExtArr");

      while (list(,$val) = each($extArr))
       {
        if (preg_match("'^$val$'i",$ext))
         {
          return(true);
         }
       }
     }
    else
     {
      return(true);
     }

      //urls with a / at the end are ever in extensionslist!
     if (substr($url,-1) == '/'){return(true);}

    return($retFlag);
   }


  function    pageExtension      ($url)
   {
    $ext = "";
    $domain = $this->getDomain($url);

    $onlyURL = preg_replace("|^$domain|","",$url);
    $onlyURL = preg_replace("/\?.+/","",$onlyURL);

    if (preg_match("/.+\.(.*)$/",$onlyURL,$arr)) { $ext = $arr[1]; }

    //TMP
    //$this->log->notice("Extension in $url =>$ext");
    return($ext);
   }


  function    validURL           ($url,
                                  &$notParseReason)
   {
    $notParseReason = "";
    $retFlag = false;

    if ($this->inBlackList($url))       { $notParseReason = "url in black list"; }
    if ($notParseReason == "")
     {
      if (!$this->inExtensionList($url)) { $notParseReason = "not in extension list"; }
     }
    if ($notParseReason == "")
     {
      if ($this->isDirectoryIndexSorting($url))
                                        { $notParseReason = "this is dir another sort type"; }
     }

    if ((!$this->inBlackList($url))    &&
        ($this->inExtensionList($url)) &&
        (!$this->isDirectoryIndexSorting($url)))
     {
      $retFlag = true;
     }

    return($retFlag);
   }


  function    isDirectoryIndexSorting
                                 ($url)
   {
    if (preg_match("/\/\?N=A$/",$url)) { return(true); }
    if (preg_match("/\/\?N=D$/",$url)) { return(true); }
    if (preg_match("/\/\?M=A$/",$url)) { return(true); }
    if (preg_match("/\/\?M=D$/",$url)) { return(true); }
    if (preg_match("/\/\?S=A$/",$url)) { return(true); }
    if (preg_match("/\/\?S=D$/",$url)) { return(true); }
    if (preg_match("/\/\?D=A$/",$url)) { return(true); }
    if (preg_match("/\/\?D=D$/",$url)) { return(true); }

    return(false);
   }


  function    updateStruct       ()
   {
    $query = "DELETE FROM $this->structTableName";

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());

     }
    else
     {
      $query = "SELECT URL from $this->DBTableName";

      $result = mysql_query($query,$this->link);

      if ($result)
       {
        while (list($elem) = mysql_fetch_row($result))
         {
          $this->addURLtoStruct($elem);
         }
       }
      else
       {
        $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
        $this->log->error($this->errMess());
       }
     }
   }


  function    optimizeMainTable  ()
   {
    $query = "
    REPAIR TABLE $this->DBTableName;
    OPTIMIZE TABLE $this->DBTableName;
    FLUSH TABLE $this->DBTableName;   
    ";

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
   }


  function    removeUrlsStartedFrom
                                 ($urlPart)
   {
    $query = "DELETE FROM $this->DBTableName WHERE URL LIKE '$urlPart%'";

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
    else
     {
      $retValue = mysql_affected_rows($this->link);
      $this->updateStruct();
     }

    return($retValue);
   }


              // Update path structur basing on $url

  function    addURLtoStruct     ($url)
   {
    $pairs = $this->getPairArray($url);

    while (list(,$val) = each($pairs))
     {
      $query =<<<END
         REPLACE into $this->structTableName SET
                   own   = '$val[0]',
                   child = '$val[1]'
END;

      if (!mysql_query($query,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
        $this->log->error($this->errMess());

       }
     }
   }


  function    getPairArray       ($url)
   {
    global         $SpidrEngineConst;

    $parts = parse_url($url);
    $arr = array();

    $elem = $parts["scheme"]."://".$parts["host"];
    if (isset($parts["port"])) { $elem.= ":".$parts[port]; }
    $arr[] = $elem;

    //if (preg_match("'/\s*$'",$url)) { $parts["path"] .= "/"; }

    $pathes = split("/",$parts["path"]);

    while (list(,$val) = each($pathes))
     {
      if ($val != "") { $arr[] = $val; }
     }

    if (count($arr) > 1) { array_pop($arr); }

    return($this->getAllStructPairs($arr));
   }


              
               /* Input is an array of elements (let it be as follows:
                *                                       [0] =>'first',
                *                                      [1] =>'second',
                *                                      [3] => 'third')
                * The output will be:
                * [0] => array(''     , 'first')
                * [1] => array('first', 'second')
                * [2] => array('second','third')
                */
              


  function    getAllStructPairs  ($arr)
   {
    $retArr = array();

    if (count($arr) >= 1) { $retArr[] = array('',$arr[0]); }

    if (count($arr) >= 2)
     {
      $prefix = "";

      for ($i = 1;$i < count($arr);$i++)
       {
        $retArr[$i] = array($arr[$i - 1],$arr[$i]);

        if ($prefix != "") { $retArr[$i][0] = $prefix.$retArr[$i][0]; }

        $prefix .= $arr[$i - 1]."/";
       }
     }



    return($retArr);
   }


  function    urlParsedYet       ($url)
   {
    $retFlag = false;

    $url_ = addslashes($url);

    $query =<<<END
         SELECT URL from __urls where URL = "$url_";
END;

    $result = mysql_query($query,$this->link);

    if (!$result)
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->setErrMess());
     }
    else
     {
      if (mysql_num_rows($result) > 0)
       {
        $this->log->notice("$url already parsed");
        $retFlag = true;
       }
     }

    return($retFlag);
   }

  function    _createParsedUrlsTmpTable
                                 ()
   {

    $query =<<<END
    CREATE TEMPORARY TABLE IF NOT EXISTS __urls
        (
        URL                  CHAR(255) NOT NULL,
        UNIQUE KEY          (URL)
        );
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->setErrMess());
     }
   }


  function    tmpTableAddUrl     ($url)
   {
    $url_ = addslashes($url);
    $query =<<<END
      REPLACE INTO __urls SET
          URL          = "$url_";
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->setErrMess());
     }
   }

 }

	//put out a message to the browser, not to the logfile
function      outMessage         ($mess = "")
 {
  global      $outInfoWeb;

  if ($outInfoWeb) { $mess .= "<br>\n"; }
  else             { $mess .= "\n";   }

  echo "$mess";
  flush();					//put out the message currently and did not wait
 }


function      getNextFileName    ()
 {
  static           $num = 0;

  return("log/file".$num++.".htm");
 }

function      _debugSaveBuff     ($url,
                                  $lines,
                                  $buf)
 {
  $fp = fopen(getNextFileName(),"w");
  fwrite($fp,"[url] = $url\n\n");
  fwrite($fp,join("",$lines)."\n\n");
  fwrite($fp,$buf);
  fclose($fp);
 }

?>

