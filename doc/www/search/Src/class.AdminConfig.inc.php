<?php
/*      

        © BY http://phpMySearch.web4.hm

        */
        
$cArrayItemsDelimiter = "<>";

              // Returns a copy of AdminConfig class

function     &AdminConfigInstance($extSource = "DATABASE")
 {
  static $fSingleton;

  if (!isset($fSingleton))
   {
    $fSingleton = new AdminConfig($extSource);

    register_shutdown_function("AdminConfigDone");
   }

  return($fSingleton);
 }
              // Automatically calls destructor

function      AdminConfigDone    ()
 {
  $inst = &AdminConfigInstance();
  $inst->done();
 }
         
          /*
           * Copy of this class should be only obtained by invoking
           * AdminConfigInstance() function
           *
           *  $instance = &AdminConfigInstance();
           *  $prm = $instance->param("param");
           */
         


class    AdminConfig extends Config
 {
              // Constructor

  function    AdminConfig        ($extSource = "")
   {
    Config::Config($extSource);
    $this->logReplaceOutFlag = true;
   }

         
          /*
           * Load setting from outer soure
           * !!! Invoked by constructor
           */
         


  function    selfLoad           ($extSource = "")
   {
    global         $DBName;
    global         $DBUser;
    global         $DBPassword;
    global         $DBHost;
    global	   $HTTP_ENV_VARS;
    global         $HTTP_SERVER_VARS;
    
    $a = explode ("admin.php", $HTTP_SERVER_VARS["PATH_TRANSLATED"]);
    $Path = $a[0];

    $this->hash = array
     (
      
       /*
        * Database setting
        * All this settings are required
        */

      "DBName"               => $DBName,
      "DBUser"               => $DBUser,
      "DBPassword"           => $DBPassword,
      "DBHost"               => $DBHost,

      "ProxyActive"             => false,
      "ProxyHost"               => "none",
      "ProxyUser"               => "User:Password",

      "DBMainTableName"        => "phpMySearch_Pages",
      "DBSettingsTableName"    => "phpMySearch_settings",
      "DBStructTableName"      => "phpMySearch_struct",
      "DBSpiderStateTableName" => "phpMySearch_spider",

      "adminLogin"           => "admin",
      "adminPassword"        => "admin",

      "adminSessionLong"     => 20, // Not in base

      "parsingExtArr"        => array("htm",
                                      "html",
                                      "txt",
                                      "ini",
                                      "conf",
                                      "sql",
                                      "log",
                                      "php",
                                      "php3",
                                      "cgi",
                                      "pl",
                                      "cfm",
                                      "pdf",
                                      "doc",
                                      "xls",                                                                            
                                      "asp",
                                      "jsp",
                                      "shtml",
                                      "phtml"),

      "startURLs"            => array("http://".$HTTP_SERVER_VARS["HTTP_HOST"]),
      "urlRetrieveNumber"    => 5,
      "blackList"            => array("spider.php",
      					"(Empty Reference!)"),

      "searchDeep"           => 5, 
                             /*
                              * Search depth
                              *
                              * 0 - don't follow any links
                              * 1 - follow links only from the first page
                              */

      "outRefsToPage" => 5,
                             /*
                              * Number of found links per page
                              */

      "maxPageRef"    => 10,
                             /*
                              * Maximum number of pages allowed for direct access
                              * if 5 the output will be
                              * (<< 1 2 3 4 5 >>>)
                              * (<< 6 7 8 9 10 >>>) ...
                              */

      "searchEngineLogFileName"    => $Path."log/search.log",
      "spiderEngineLogFileName"    => $Path."log/spider.log",
      "adminConfigLogFileName"     => $Path."log/admin.log",

      "templatesPath"              => $Path."templates/Aqua",


      "PDFConverterURL"            => "http://access.adobe.com/perl/convertPDF.pl",
      "PDFConverterVarName"        => "url",
      "PDFConverterVarTransMethod" => "GET",

      "DOCConverterURL"            => "http://localhost/converter/DOCConverter.php",
      "DOCConverterVarName"        => "url",
      "DOCConverterVarTransMethod" => "GET",
      
      "XLSConverterURL"            => "http://localhost/converter/XLSConverter.php",
      "XLSConverterVarName"        => "url",
      "XLSConverterVarTransMethod" => "GET",
                  
       /*
        * When spider is started parse all pages regardless of expiration date
        * of the page (true)
        * Parse only expired pages (false)
        */

      "spiderEngineReparseAll"     => true,
      "spiderAutoRestart"          => true,

      "spiderOnlySetUpDomain"      => true,

      "spiderTimeStart"            => "21:00:00",
      "spiderStartDaysPeriod"      => 7,

      "spiderMaxDescriptionLength" => 250,
      "spiderMaxAuthorLength"      => 50,
      "spiderMaxKeywordLength"     => 250,

      "phpFullPath"                => "php"

      //"adminUpdateDBByDeleteURL"   => true
     );



    $this->log = &LogInstance($this->param("adminConfigLogFileName"));

    if ($extSource != "")
     {
      if ($extSource == "DATABASE")
       {
        $this->loadFromDB();
       }
      else
       {
        $this->setErrMess("This external source ('$extSource') not supported");

        $this->log->error($this->errMess());
       }
     }
   }


  function    setParam           ($key,
                                  $value)
   {
    global         $cArrayItemsDelimiter;
    $old = $this->param($key);
    $new = $value;

    if (gettype($old) == "array") { $old = join($cArrayItemsDelimiter,$old); }
    if (gettype($new) == "array") { $new = join($cArrayItemsDelimiter,$new); }

    Config::setParam($key,$value);

    if (($old != $new) && ($this->logReplaceOutFlag))
     {
      $this->log->notice("Parameter ($key) is replaced from '$old' to '$new'");
     }
   }


  function    selfSave           ($extSource = "DATABASE")
   {
    $this->setErrMess("");

    $this->log->notice("settings selfSave to '$extSource' executed");

    if ($extSource != "")
     {
      if ($extSource == "DATABASE")
       {
        $this->DBSetDefault($this->param("DBSettingsTableName"));
       }
      else
       {
        $this->setErrMess("This external source ('$extSource') not supported");

        $this->log->error($this->errMess());
       }
     }
   }


  function    loadFromDB         ()
   {
    global         $cArrayItemsDelimiter;
    $this->logReplaceOutFlag = false;

    $this->setErrMess("");
    $errMessPrefix = "DB Error: ";
    $host   = $this->param("DBHost");
    $user   = $this->param("DBUser");
    $pswd   = $this->param("DBPassword");
    $DBName = $this->param("DBName");

    $const = &ConstInstance();
    $tableStruct = $const->settingsTableStruct();
    $tableName   = $this->param("DBSettingsTableName");

    $this->link = mysql_connect($host,$user,$pswd);

    if ($this->link > 0)
     {
      if (!mysql_select_db($DBName,$this->link))
       {
        $this->setErrMess($errMessPrefix."Select DB error: $DBName");
        $this->log->error($this->errMess());

       }
     }
    else
     {

      $this->setErrMess($errMessPrefix."Could not connect to DataBase: $DBName");

      $this->log->error($this->errMess());
     }

    if ($this->errMess() == "")
     {
      $this->DBTableCreate($tableName,$tableStruct);
     }

    if ($this->errMess() == "")
     {
      $query = "SELECT * from $tableName;";
      $result = mysql_query($query,$this->link);

      if (!$result)
       {

        $this->setErrMess("Table selection data error");
        $this->log->error("Bad query: $query => (".mysql_error($this->link).")");
       }
      else
       {
        // If record is selected, put data in properties

        if ($arr = mysql_fetch_assoc($result))
         {
          while (list($key,$val) = each($arr))
           {
            if ($key != "recNo")
             {
              $type = gettype($this->param($key));


              switch ($type)
               {
                case ("array"):
                 {
                  if (trim($val) != "")
                   {
                    $this->setParam($key,split($cArrayItemsDelimiter,$val));
                   }
                  else
                   {
                    $this->setParam($key,array());
                   }
                 } break;

                case ("boolean"):
                 {
                  if ($val == 0) { $this->setParam($key,false); }
                  else           { $this->setParam($key,true);  }
                 } break;

                default:
                 {
                  $this->setParam($key,$val);
                 }
               }
             }
           }


         }
        // Otherwise store default settings to the table

        else
         {
          $this->DBSetDefault($tableName);
         }
       }
     }

    $this->logReplaceOutFlag = true;
   }


  function    DBSetDefault       ($tableName)
   {
    global         $cArrayItemsDelimiter;

    $this->setErrMess("");

    $query = "REPLACE INTO $tableName SET recNo = 1,";

    $paramNames = $this->paramNames();

    while (list(,$key) = each($paramNames))
     {
      $val = $this->param($key);
      $type = gettype($val);

      if     (($type == "integer") ||
              ($type == "double"))
       {
        $query .= "$key = $val,";
       }
      elseif ($type == "array")
       {
        $query .= "$key = \"".join($cArrayItemsDelimiter,$val)."\",";
       }
      elseif ($type == "boolean")
       {
        if   ($val) { $query .= "$key = 1,"; }
        else        { $query .= "$key = 0,"; }
       }
      else
       {
        $query .= "$key = \"$val\",";
       }
     }

    $query = preg_replace("/,$/","",$query);

    if (!mysql_query($query,$this->link))
     {

      $this->setErrMess("Table creation error");
      $this->log->error("Bad query: $query => (".mysql_error($this->link).")");
     }
   }


  function    DBTableCreate      ( $tableName,
                                   $tableStruct)
   {
    $this->setErrMess("");

    $query =<<<END
       CREATE TABLE IF NOT EXISTS $tableName ($tableStruct)
END;

    if (!mysql_query($query,$this->link))
     {

      $this->setErrMess("Table creation error");
      $this->log->error("Bad query: $query => (".mysql_error($this->link).")");
     }
   }


  function    done               ()
   {
    if (isset($this->link)) { mysql_close($this->link); }
   }

  var    $link;
  var    $log;
  var    $logReplaceOutFlag;
 }

?>
