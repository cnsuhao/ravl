<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

function      &SearchEngineInstance
                                 ( $DBName,
                                   $host,
                                   $user,
                                   $password,
                                  &$errMess)
 {
  static      $fSingleton;

  if (!isset($fSingleton))
   {
    $fSingleton = new SearchEngine($DBName,$host,$user,$password,$errMess);

    register_shutdown_function("SearchEngineInstanceDone");
   }

  return($fSingleton);
 }


function      SearchEngineInstanceDone
                                 ()
 {
  $errMess = "";
  $inst = &SearchEngineInstance("","","","",&$errMess);
  $inst->done();
 }



 /* Class works as follows.
  * Query string processing description
  * Let's have an expression 'table AND not chair OR bookcase'
  * The expression is transfered in reverse polish notation
  *    table     (operand)
  *    not chair (operand)
  *    AND       (operator)
  *    bookcase  (operand)
  *    OR        (operator)
  *
  * Than we go thrugh array if we meet operand we run query for operand
  * For our example we run query for "table", than "not chair" and put results
  * in temporary table with generated unique name. The name of the table is stored
  * in the operand stack
  *
  * If operator is met script will take from the stack name of two tables and will
  * perform operation on them.
  *
  * If OR is met script produces union of those 2 tables
  *
  * If AND is met script performs table intersection
  *
  * The result is stored in temporary table with unique name.
  * Processed temporary tables are droped.
  *
  * After processing whole array and taking all tokens and performing
  * all operations in stack we'll have the final temporary table with
  * result set for user query.
  * Than we' query the above mantioned table and store result to OutTmp
  * OutTmp has a little differ structure than all other temporary tables.
  * It is key is (inKeywordsFlag,relevance,URL) and not URL.
  * As a result we'll get records in required order.
  */



class    SearchEngine
 {
  // === public ===
  function    SearchEngine       ( $DBName,
                                   $host,
                                   $user,
                                   $password,
                                  &$errMess)
   {
    global    $SearchEngineConst;

    $errMess = "";
    $errMessPrefix = $SearchEngineConst['DataBaseErrPrefixMsg'];

    $this->adminConfig = &AdminConfigInstance();
    $this->log = &LogInstance($this->adminConfig->param("searchEngineLogFileName"));
    $this->log->start();

    $this->const  = &ConstInstance();
    $this->fMainTmpTableName     = $this->const->mainTmpTableName();
    $this->fOutTmpTableName      = $this->const->outTmpTableName();
    $this->fMainTableName        = $this->adminConfig->param("DBMainTableName");
    $this->fMainTmpTableStruct   = $this->const->tmpTableStruct();
    $this->fOutTmpTableStruct    = $this->const->outTmpTableStruct();

    $this->link = mysql_connect($host,$user,$password);

    if ($this->link > 0)
     {
      if (!mysql_select_db($DBName,$this->link))
       {
        $errMess = $errMessPrefix.sprintf($SearchEngineConst['SelectDBErrMsg'],$DBName);
        $this->log->error($errMess);

       }
     }
    else
     {

      $errMess = $errMessPrefix.sprintf($SearchEngineConst['DataBaseConnectionErrMsg'],$DBName);

      $this->log->error($errMess);
     }

    // Purge expired records

    if ($errMess == "")
     {
      $query = <<<END
      DELETE FROM $this->fMainTableName
             WHERE expiresFlag = 1 AND expiresDate < CURDATE()
END;

      if (!mysql_query($query,$this->link))
       {
       	//pp
       	$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
        $errMess = $SearchEngineConst['DBExpiredRecodsDeletedMsg'];

        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }
   }


	function    done               ()
	{
		$this->log->end();
	}
              
               /*
                * parse expression for $expr and return set of resulting
                * records.
                * Each element is represented with hash
                * field name => value
                *
                * If $beginRow and $rows parameters specified than only
                * $row of strings returned
                */
              
  function    parseExpression    ( $expr = "",
                                   $path = "",
                                  &$addInfo,
                                  &$errMess,
                                   $beginRow = 0,
                                   $rows     = 0)
   {
    global         $SearchEngineConst;
    global         $SCRIPT_NAME;

    //If will need restart the spider it will be run in background

    $spiderState = new SpiderState();

    if ($spiderState->needRestartSpider())
     {
      if ($this->adminConfig->param("spiderAutoRestart"))
       {
        $spiderState->restart();
       }
     }

    $spiderState->done();

    // if string $errMess is not empty than an error occured
    $errMess = "";
    
    $errMessPrefix = $SearchEngineConst['ExpressionParsingErrPrefixMsg'];

    $this->log->notice(sprintf($SearchEngineConst['ExpressionParsingStartMsg'],$expr));
    $result = array();

    //remove spaces from beginning and from the end
    $expr = trim($expr);

    if ($expr != "")
     {
      $this->prepareTmpTables($errMess);

      if ($errMess == "")
       {
        $inst = &ExpressionPrepocessorInstance($expr);
        $expr = $inst->processedStr();

        $revPol = new RevPolNotation($expr);

        $tokenArr = $revPol->get($errMess);
       }

      $operandStack = array();

      while (($errMess == "") &&
             (list(,$val) = each($tokenArr)))
       {
        $this->log->notice(sprintf($SearchEngineConst['TokenInSequenceMsg'],$val["item"]));

        if      ($val["type"] == "operand")
         {
          $this->log->notice(sprintf($SearchEngineConst['OperandPushedToArrayMsg'],$val["item"]));

          $tableName = getOrigTmpName();

          if ($val["UnaryOp"] == "")
           {
            if (preg_match("/^\{(.*)\}$/",$val["item"],$arr))
             {
              $val["item"] = $arr[1];
              $this->getSelection($val["item"],$path,"folder",$tableName,$errMess);
             }
            else
             {
              $this->getSelection($val["item"],$path,"word",$tableName,$errMess);
             }
           }
          else
           {
            if (preg_match("/^\{(.*)\}$/",$val["item"],$arr))
             {
              $val["item"] = $arr[1];
              $this->getNotSelection($val["item"],$path,"folder",$tableName,$errMess);
             }
            else
             {
              $this->getNotSelection($val["item"],$path,"word",$tableName,$errMess);
             }
           }

          if ($errMess == "")
           {
            $val["item"] = $tableName;

            array_push($operandStack,$val);
           }
         }
        else if ($val["type"] == "operator")
         {
          $op1 = array_pop($operandStack);
          $op2 = array_pop($operandStack);
	
          if (($op1 == NULL) || ($op2 == NULL))
           {
            $errMess = $errMessPrefix.$SearchEngineConst['EmptyOperandStackErrMsg'];
            $this->log->error($errMess);

           }

          if ($errMess == "")
           {
            $this->log->notice(sprintf($SearchEngineConst['ExecuteOperationMsg'],$val[item]));

            $this->log->notice(sprintf($SearchEngineConst['OperationInfoMsg'],$op1,$val[item],$op2));
            
            $this->executeOperation($op1,$op2,$val["item"],&$resOperand,$errMess);
           }

          if ($errMess == "")
           {
            $this->log->notice(sprintf($SearchEngineConst['OperationResultMsg'],$resOperand[item]));

            array_push($operandStack,$resOperand);
           }
         }
        else
         {
          $errMess = $errMessPrefix.sprintf($SearchEngineConst['TokenTypeUnknownErrMsg'],$val[type]);

          $this->log->error($errMess);
         }
       }

      if ($errMess == "")
       {
        $res = array_pop($operandStack);
        $this->log->notice($SearchEngineConst['StackPopResultValueMsg']);

        if ($res == NULL) { $errMess = $errMessPrefix.$SearchEngineConst['EmptyOperandStackErrMsg'];
                            $this->log->fatal($errMess);

                          }
       }

      if ($errMess == "")
       {
        $result = $this->getResultArray($res["item"],&$addInfo,$beginRow,$rows,$errMess);
       }

	//PP
      //$this->dropTmpTables();
     }
    else
     {
      $this->log->warning($SearchEngineConst['ExpressionEmptyErrMsg']);

     }

    return($result);
   }

              
               /*
                * Returns all subfolders of folder $dir
                * if $dir = '' returns all subfolders of root folders
                */
              


  function    getSubDirs         ( $dir,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $retArray = array();
    $structTableName = $this->adminConfig->param("DBStructTableName");

    $query =<<<END
         SELECT HIGH_PRIORITY child FROM $structTableName WHERE own = "$dir"
END;

    $result = mysql_query($query,$this->link);
	//PP
	$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
	
    if ($result)
     {
      while (list($elem) = mysql_fetch_row($result))
       {
        $retArray[] = $elem;
       }
     }
    else
     {

      $errMess = $SearchEngineConst['DBSelectionErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    return($retArray);
   }


  // === private ===

  function    executeOperation   ( $op1,
                                   $op2,
                                   $operator,
                                  &$resOp,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $resTableName = getOrigTmpName();

    $resOp = array
     (
      "item"    => $resTableName,
      "UnaryOp" => "",
      "type"    => "operand"
     );

    switch ($operator)
     {
      case ("+"):
       {
        $this->orResult($op1["item"],$op2["item"],$resTableName,$errMess);
       } break;

      case ("*"):
       {
        $this->andResult($op1["item"],$op2["item"],$resTableName,$errMess);
       } break;

      default:
       {

        $errMess = sprintf($SearchEngineConst['UndefinedOperatorErrMsg'],$operator);
        $this->log->error($errMess);
       }
     }
   }

              
               /* Returns an array of serach results. Each elements are
                * represented by hash
                * field name => value
                */
              


  function    getResultArray     ( $resTabelName,
                                  &$addInfo,
                                   $beginRow,
                                   $rows,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $errMess = "";

    $this->prepareOutResult($resTabelName,&$addInfo,$beginRow,$rows,$errMess);

    if ($errMess == "")
     {
      $retArray = array();

      $query = "SELECT HIGH_PRIORITY * from $this->fOutTmpTableName";

	$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $result = mysql_query($query,$this->link);

      if (!$result)
       {

        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_num_rows($result)));

      $index = 0;

      while ($arr = mysql_fetch_assoc($result))
       {
        $elem = array();

        while (list($key,$val) = each($arr))
         {
          $elem[$key] = $val;
         }

        $retArray[$index] = $elem;
        $index++;
       }

      mysql_free_result($result);
     }

    return($retArray);
   }

              
               /*
                * Perform operation AND for tables and store result
                * to the table $resTableName, which is created by
                * this function
                */
              


  function    andResult          ( $firstTableName,
                                   $secondTableName,
                                   $resTableName,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $errMess = "";

    $this->log->notice($SearchEngineConst['ANDOperationMsg']);

    $query = "
    CREATE TEMPORARY TABLE IF NOT EXISTS $resTableName ($this->fMainTmpTableStruct)";


    if (!mysql_query($query,$this->link))
     {
      $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];

      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    if ($errMess == "")
     {
      $query =<<<END
      INSERT INTO $resTableName
                  SELECT HIGH_PRIORITY
                        $firstTableName.URL,
                        $firstTableName.pageDate,
                        $firstTableName.expiresDate,
                        $firstTableName.title,
                        $firstTableName.description,
                        $firstTableName.keywords,
                        $firstTableName.author,
                        $firstTableName.replyTo,
                        $firstTableName.publisher,
                        $firstTableName.copyright,
                        $firstTableName.contentLanguage,
                        $firstTableName.pageTopic,
                        $firstTableName.pageType,
                        $firstTableName.abstract,
                        $firstTableName.classification,
                        $firstTableName.body_1,
                        $firstTableName.body_2,
                        $firstTableName.expiresFlag,
                        $firstTableName.RelevanceUrl
                        + $secondTableName.RelevanceUrl
                        - $firstTableName.RelevanceUrl * $secondTableName.RelevanceUrl,
                        $firstTableName.RelevancePUrl
                        + $secondTableName.RelevancePUrl
                        - $firstTableName.RelevancePUrl * $secondTableName.RelevancePUrl,
                        $firstTableName.RelevanceTitle
                        + $secondTableName.RelevanceTitle
                        - $firstTableName.RelevanceTitle * $secondTableName.RelevanceTitle,
                        $firstTableName.RelevanceBody
                        + $secondTableName.RelevanceBody
                        - $firstTableName.RelevanceBody * $secondTableName.RelevanceBody
                  	FROM $firstTableName, $secondTableName
                        WHERE $firstTableName.URL = $secondTableName.URL 
                        
END;
//PP
      $result = mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      if (!$result)
       {

        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_affected_rows($this->link)));
     }

    //PP$query = "DROP TABLE IF EXISTS $firstTableName";
    mysql_query($query,$this->link);
    //PP
    $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));

    //PP$query = "DROP TABLE IF EXISTS $secondTableName";
    mysql_query($query,$this->link);
    //PP
    $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
   }



              
		/*
		* Perform OR operation for tables and store result
		* to the table $resTableName, which is created by
		* this function
		*/

  function    orResult           ( $firstTableName,
                                   $secondTableName,
                                   $resTableName,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $errMess = "";

    $this->log->notice($SearchEngineConst['OROperationMsg']);

    $query = "
    CREATE TEMPORARY TABLE IF NOT EXISTS $resTableName ($this->fMainTmpTableStruct)";

    if (!mysql_query($query,$this->link))
     {

      $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    if ($errMess == "")
     {

      $query = "

	INSERT HIGH_PRIORITY INTO $resTableName SELECT * from $firstTableName

      ";

      if (!mysql_query($query,$this->link))
       {
        $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {

      $query =<<<END

	REPLACE INTO $resTableName SELECT HIGH_PRIORITY * FROM $secondTableName;
END;

	$result = mysql_query($query,$this->link);
	
	$query =<<<END
	REPLACE INTO $resTableName
		SELECT HIGH_PRIORITY 
		$secondTableName.URL,
		$secondTableName.pageDate,
		$secondTableName.expiresDate,
		$secondTableName.title,
		$secondTableName.description,
		$secondTableName.keywords,
		$secondTableName.author,
		$secondTableName.replyTo,
		$secondTableName.publisher,
		$secondTableName.copyright,
		$secondTableName.contentLanguage,
		$secondTableName.pageTopic,
		$secondTableName.pageType,
		$secondTableName.abstract,
		$secondTableName.classification,
		$secondTableName.body_1,
		$secondTableName.body_2,
		$secondTableName.expiresFlag,
		$secondTableName.RelevanceUrl + $firstTableName.RelevanceUrl,
		$secondTableName.RelevancePUrl + $firstTableName.RelevancePUrl,
		$secondTableName.RelevanceTitle + $firstTableName.RelevanceTitle,
		$secondTableName.RelevanceBody + $firstTableName.RelevanceBody
	FROM $secondTableName, $firstTableName
	WHERE $secondTableName.URL = $firstTableName.URL
	##OR2

END;

      $result = mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      if (!$result)
       {
        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_affected_rows($this->link)));
     }

    //PP$query = "DROP TABLE IF EXISTS $firstTableName";
    mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
    //PP$query = "DROP TABLE IF EXISTS $secondTableName";
    mysql_query($query,$this->link);
    
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
   }


              
               /*
                * Perform query by expression element and store result in
                * temporary table $resTableName
                */
              


  function    getSelection       ( $elem,
                                   $path,
                                   $itemType,
                                   $resTableName,
                                  &$errMess)
   {
    global         $SearchEngineConst;



    $errMess = "";

    $this->log->notice($SearchEngineConst['GetSelectionExecutedMsg']);

    $query = "
    CREATE TEMPORARY TABLE IF NOT EXISTS $resTableName ($this->fMainTmpTableStruct)";


    if (!mysql_query($query,$this->link))
     {
      $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    if ($errMess == "")
     {
      if ($itemType == "word")
       {
        $query = <<<END
	INSERT INTO $resTableName
	SELECT *,
	IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,1,0) AS RelevanceUrl,
	IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))),0) AS RelevancePUrl,
	MATCH(title,title) AGAINST("$elem") + IF ( LOCATE( UPPER("$elem") , UPPER( keywords ) )  > 0,1,0 ) AS RelevanceTitle,
	MATCH(body_1,body_2) AGAINST("$elem") AS RelevanceBody
	FROM $this->fMainTableName
	WHERE URL REGEXP '^$path' AND 
	(
	IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,1,0)  > 0 OR
	IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))),0)  > 0 OR
	MATCH(title,title) AGAINST("$elem") + IF ( LOCATE( UPPER("$elem") , UPPER( keywords ) )  > 0,1,0 ) > 0 OR
	MATCH(body_1,body_2) AGAINST("$elem") > 0
	)
	AND expiresFlag = 0

END;
       }
      else
       {
        if ($path != "") { $regexp = "^$path/$elem(/|$)"; }
        else             { $regexp = "^$elem(/|$)"; }

        $query = <<<END
        INSERT INTO $resTableName
               SELECT HIGH_PRIORITY *,
                      0 AS RelevanceUrl,
                      0 AS RelevancePUrl,
                      0 AS RelevanceTitle,
                      0 AS RelevanceBody
               FROM $this->fMainTableName
               WHERE URL REGEXP "$regexp"
              
END;
       }



      $result = mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      if (!$result)
       {

        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_affected_rows($this->link)));
     }
   }

              
               /*
                * Perform query by expression element. In the result set
                * will be stored records that doesn't contain this element
                * (not $elem). Saves the result in temporary table
                * $resTableName
                */
              


  function    getNotSelection    ( $elem,
                                   $path,
                                   $itemType,
                                   $resTableName,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $errMess = "";

    $this->log->notice($SearchEngineConst['GetNotSelectionExecutedMsg']);

    $query = "
    CREATE TEMPORARY TABLE IF NOT EXISTS $resTableName ($this->fMainTmpTableStruct)";

    if (!mysql_query($query,$this->link))
     {
      $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    if ($errMess == "")
     {
      if ($itemType == "word")
       {
        $query = <<<END
        INSERT INTO $resTableName
               SELECT HIGH_PRIORITY *,
                      0 AS RelevanceUrl,
                      0 AS RelevancePUrl,
                      0 AS RelevanceTitle,
                      0 AS RelevanceBody
               FROM $this->fMainTableName
               WHERE URL REGEXP '^$path' AND
			(
			IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,1,0)  = 0 AND
			IF (LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))) > 0,LOCATE(UPPER("$elem"),UPPER(CONCAT(url,' ',' '))),0)  = 0 AND
			MATCH(title,title) AGAINST("$elem") + IF ( LOCATE( UPPER("$elem") , UPPER( keywords ) )  > 0,1,0 ) = 0 AND
			MATCH(body_1,body_2) AGAINST("$elem") = 0
			)
			AND expiresFlag = 0
END;

       }
      else
       {
        if ($path != "") { $regexp = "^$path/$elem(/|$)"; }
        else             { $regexp = "^$elem(/|$)"; }

        $query = <<<END
        INSERT INTO $resTableName
               SELECT HIGH_PRIORITY *,
                      0 AS RelevanceUrl,
                      0 AS RelevancePUrl,
                      0 AS RelevanceTitle,
                      0 AS RelevanceBody
               FROM $this->fMainTableName
               WHERE URL REGEXP     "^$path" AND
                     URL NOT REGEXP "$regexp"
END;
       }

      $result = mysql_query($query,$this->link);

      if (!$result)
       {

        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_affected_rows($this->link)));
     }
   }


              // Prepare table for output

  function    prepareOutResult   ( $resTableName,
                                  &$addInfo,
                                   $beginRow,
                                   $rows,
                                  &$errMess)
   {
    global         $SearchEngineConst;

    $errMess = "";

    $this->log->notice($SearchEngineConst['PrepareOutResultMsg']);

    $query = <<<END
         INSERT INTO $this->fOutTmpTableName
                SELECT HIGH_PRIORITY
			URL,
			pageDate,
			expiresDate,
			keywords,
			title,
			description,
			author,
			replyTo,
			publisher,
			copyright,
			contentLanguage,
			pageTopic,
			pageType,
			abstract,
			classification,
			body_1,
			body_2,
			RelevanceUrl,
			RelevancePUrl,
			RelevanceTitle,
			RelevanceBody
                FROM $resTableName
                ORDER BY RelevanceUrl DESC, RelevancePUrl ASC, RelevanceTitle DESC, RelevanceBody DESC
END;

    if ($rows != 0) { $query .= " LIMIT $beginRow,$rows"; }

    $result = mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
    if (!$result)
     {

      $errMess = $SearchEngineConst['DBSelectionErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }

    if ($errMess == "")
     {
      $rowsNum = mysql_affected_rows($this->link);
      $this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      $this->log->notice(sprintf($SearchEngineConst['LogRowCountInfoMsg'],mysql_affected_rows($this->link)));

      $query = "SELECT HIGH_PRIORITY COUNT(*) as count FROM $resTableName";
      $result = mysql_query($query,$this->link);
//PP
$this->log->notice(sprintf($SearchEngineConst['LogQueryInfoMsg'],$query));
      if (!$result)
       {

        $errMess = $SearchEngineConst['DBSelectionErrMsg'];
        $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
       }
     }

    if ($errMess == "")
     {
      $countArr = mysql_fetch_assoc($result);
      mysql_free_result($result);

      $addInfo["pages"] = $countArr["count"];
     }
   }

              
               /*
                * Create necessary temporary tables
                * Befor perform operation
                *    $this->prepareTmpTables();
                *    .....
                *    $this->dropTmpTables();
                */
              


  function    prepareTmpTables   (&$errMess)
   {
    global         $SearchEngineConst;

	$this->dropTmpTables();

    $query = "
    CREATE TEMPORARY TABLE $this->fOutTmpTableName ($this->fOutTmpTableStruct)";

    if (!mysql_query($query,$this->link))
     {

      $errMess = $SearchEngineConst['TmpTableCreationErrorErrMsg'];
      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }
   }
              // Drop temporary tables

  function    dropTmpTables      ()
   {
    global         $SearchEngineConst;

    $query = "DROP TABLE IF EXISTS $this->fOutTmpTableName";

    if (!mysql_query($query,$this->link))
     {

      $this->log->error(sprintf($SearchEngineConst['BadQueryErrMsg'],$query,mysql_error($this->link)));
     }
   }

  var    $link;
  var    $adminConfig;
  var    $config;
  var    $fMainTmpTableName;
  var    $fOutTmpTableName;
  var    $fMainTmpTableStruct;
  var    $fOutTmpTableStruct;
  var    $fMainTableName;

  var    $log;
 }


function      getOrigTmpName     ()
 {
  static           $currNumTable = 0;

  return("tmp".$currNumTable++);
 }

?>
