<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

class    SpiderState
 {            // Constructor

  function    SpiderState        ()
   {
    global         $SpidrEngineConst;

    $this->setErrMess("");

    $this->adminConfig = &AdminConfigInstance();
    $this->DBTableName = $this->adminConfig->param("DBSpiderStateTableName");

    $this->log = &LogInstance($this->adminConfig->param("spiderEngineLogFileName"));

    $this->DBName = $this->adminConfig->param("DBName");
    $this->link = mysql_connect
                   ($this->adminConfig->param("DBHost"),
                    $this->adminConfig->param("DBUser"),
                    $this->adminConfig->param("DBPassword"));

    if ($this->link > 0)
     {
      $this->log->notice($SpidrEngineConst['DBConnectOkMsg']);

      if (!mysql_select_db($this->DBName,$this->link))
       {
        $this->setErrMess(sprintf($SpidrEngineConst['DBSelectErrMsg'],$this->DBName));
        $this->log->error($this->errMess());

       }

      if ($this->errMess() == "")
       {
        $this->createTables();
       }
     }
    else
     {

      $this->setErrMess("SpiderState: ".$SpidrEngineConst['DBConnectionErrMsg']);
      $this->log->error($this->errMess());

     }
   }
              // Destructor

  function    done               ()
   {
    if (isset($this->link)) { mysql_close($this->link); }
   }


  function    needRestartSpider  ()
   {
    global         $SpidrEngineConst;

    $time       = $this->adminConfig->param("spiderTimeStart");
    $daysPeriod = $this->adminConfig->param("spiderStartDaysPeriod");

    $query =<<<END
         SELECT count(dateTime) from $this->DBTableName
                   WHERE DATE_ADD(dateTime,INTERVAL $daysPeriod DAY) + INTERVAL TIME_TO_SEC("$time") SECOND < NOW()
END;

    $result = mysql_query($query,$this->link);




    if (!$result)
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
    else
     {

      $arr = mysql_fetch_row($result);

      if ((mysql_num_rows($result) > 0) &&
          ($arr[0] > 0))
       {
        $this->updateStateWithCurrDate();

        if ($this->errMess() == "") { return(true); }
       }
     }

    return(false);
   }
              
               /*
                * Function should be called after strting the spider
                * it refreshes spider state for its next automatic calls
                */
              


  function    updateStateWithCurrDate
                                 ()
   {
    global         $SpidrEngineConst;


    $time       = $this->adminConfig->param("spiderTimeStart");
    $daysPeriod = $this->adminConfig->param("spiderStartDaysPeriod");

    $query =<<<END
          UPDATE $this->DBTableName SET
            recNo    = 1,
            dateTime = NOW();
END;
    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
      //echo "!Error<br> ";
     }
   }


  function    setErrMess         ($errMess)
   {
    $this->errMess = $errMess;
   }


  function    errMess            ()
   {

    return($this->errMess);
   }


  function    createTables       ()
   {
    global         $SpidrEngineConst;

    $const = &ConstInstance();
    $tableStruct = $const->spiderTableStruct();

    $query =<<<END
              CREATE TABLE IF NOT EXISTS $this->DBTableName ($tableStruct);
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }

    $query =<<<END
         SELECT * from $this->DBTableName;
END;

    $result = mysql_query($query,$this->link);

    if (!$result)
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
    else
     {

      $arr = mysql_fetch_row($result);

      if (mysql_num_rows($result) == 0)
       {
        $query =<<<END
                  INSERT INTO $this->DBTableName SET
                    recNo        = 1,
                    dateTime     = NOW(),
                    nowStarted   = 0;
END;

        if (!mysql_query($query,$this->link))
         {
          $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
          $this->log->error($this->errMess());
         }
       }
     }
   }

  function    spiderStart        ()
   {
    global         $SpidrEngineConst;

    $query =<<<END
              UPDATE $this->DBTableName SET
                recNo        = 1,
                nowStarted   = 1;
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
   }

  function    spiderStopped      ()
   {
    global         $SpidrEngineConst;

    $resFlag = false;

    $query =<<<END
              SELECT nowStarted from $this->DBTableName;
END;

    $result = mysql_query($query,$this->link);

    if (!$result)
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
    else
     {
      
      $arr = mysql_fetch_row($result);

      if ((mysql_num_rows($result) > 0) &&
          ($arr[0] == 0))
       {
        $resFlag = true;
       }
     }

    return($resFlag);
   }

  function    spiderStop         ()
   {
    global         $SpidrEngineConst;

    $query =<<<END
              UPDATE $this->DBTableName SET
                recNo        = 1,
                nowStarted   = 0;
END;

    if (!mysql_query($query,$this->link))
     {
      $this->setErrMess(sprintf($SpidrEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
      $this->log->error($this->errMess());
     }
   }


  function    restart            ()
   {


    //if ($this->spiderStopped())
     {
      $this->spiderStart();
      exec($this->adminConfig->param("phpFullPath")." -q spider.php > /dev/null 2>&1 &");
     }
   }

  var    $link;
  var    $adminConfig;
  var    $DBTableName;
  var    $log;
  var    $DBName;
  var    $errMess;
 }


?>
