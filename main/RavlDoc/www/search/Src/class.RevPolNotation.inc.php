<?php
/*      

        й BY http://phpMySearch.web4.hm

        */

 /*
  * Stack item structure
  * arr["item"]    - item
  * arr["type"]    - item type (operand, operator, openBracket, closeBracket, quote)
  * arr["UnaryOp"] - unary operator before operand (for '-3' it will be '-')
  */

class RevPolNotation
 {
  // === public ===
              // Object constructor

  function    RevPolNotation     ($expression)
   {
    $this->adminConfig = &AdminConfigInstance();
    $this->log = &LogInstance($this->adminConfig->param("searchEngineLogFileName"));

    // Operator priority

    $this->fOpMap = array
     (
      '-' => 3, // Unary minus
      '+' => 1,
      '*' => 2,
      '(' => 0,
      ')' => 0
     );

    // Map of operators

    $this->fExprMap = array
     (
      "AND" => '*',
      "OR"  => '+',
      "NOT" => '-'
     );

    $this->fExpr       = " ".$expression;

    while (list($operator,$val) = each($this->fExprMap))
     {
      while (preg_match("/\s$operator(\s|$)/i",$this->fExpr))
       {
        $this->fExpr = preg_replace("/\s$operator(\s|$)/i"," $val ",$this->fExpr);
       }
     }

    $this->fQuteBeginFlag = false;
   }

              // Destructor

  function    done               ()
   {
   }

               /*
                * Get the variable to constructor
                * in reverse Polish notation
                */

  function    get                (&$errMess)
   {
    global         $RevPolNotationConst;
    $retArray    = array();
    $stack       = array();
    $oldItemType = "operator";

    $errMess = "";
    $errMessPrefix = $RevPolNotationConst['ExpressionErrorErrPrefixMsg'];

    while (($errMess == "") &&
           ($this->getNextToken(&$tok,$errMess)))
     {
      $this->log->notice(sprintf($RevPolNotationConst['LogTokenInfoMsg'],$tok["item"],$tok["type"]));

      $errMess = $this->checkValidItemOrder($tok["type"],$oldItemType);

      if ($errMess == "")
       {
        if ($tok["type"] == "closeBracket")
         {
          $item = array_pop($stack);

          while (($item != NULL) &&
                 ($item["type"] != "openBracket"))
           {
            array_push($retArray,$item);
            $item = array_pop($stack);
           }

          if ($item == NULL)
           {

            $errMess = $errMessPrefix.$RevPolNotationConst['ClosingBracketWithoutOpenErrMsg'];
            $this->log->error($errMess);

           }
         }
        elseif ($tok["type"] == "openBracket")
         {
          array_push($stack,$tok);
         }
        elseif ($tok["type"] == "operator")
         {
          if (sizeof($stack) > 0)
           {
            $item = array_pop($stack);

            while (($item != NULL) &&
                   ($this->fOpMap[$item["item"]] > $this->fOpMap[$tok["item"]]))
             {
              array_push($retArray,$item);
              $item = array_pop($stack);
             }

            if ($item != NULL) { array_push($stack,$item); }

            array_push($stack,$tok);
           }
          else
           {
            array_push($stack,$tok);
           }
         }
        else if ($tok["type"] == "operand")
         {
          array_push($retArray,$tok);
         }
        else
         {
          $errMess = $errMessPrefix.$RevPolNotationConst['TokenTypeUndefinedErrMsg'];
          $this->log->fatal($errMess);
         }
       }

      $oldItemType = $tok["type"];
     }

    if ($errMess == "")
     {
      $item = array_pop($stack);

      while ($item != NULL)
       {
        array_push($retArray,$item);
        $item = array_pop($stack);
       }
     }

    return($retArray);
   }

  // === private ===
              
               /*
                * get the next token of expression
                * token is expressed by hash with structure
                *  item    -  token
                *  type    -  token type
                *  UnaryOp -  Unary token operator, if such exists, otherwise ''
                */

  function    getNextToken       (&$tok,
                                  &$errMess)
   {
    global         $RevPolNotationConst;

    $errMess = "";
    $errMessPrefix = $RevPolNotationConst['ExpressionErrorErrPrefixMsg'];

    $tok = array("item"    => '',
                 "type"    => "operand",
                 "UnaryOp" => '');

    $this->fExpr   = ltrim ($this->fExpr);
    $this->exprLen = strlen($this->fExpr);

    if ($this->exprLen == 0) { return(false); }

    $tok["item"] = substr($this->fExpr,0,1);

            // Brackets

    if     ($this->isOpenBracket($tok["item"]))
     {
      $tok["type"] = "openBracket";
      $this->fExpr = substr($this->fExpr,1);
     }
    elseif ($this->isCloseBracket($tok["item"]))
     {
      $tok["type"] = "closeBracket";
      $this->fExpr = substr($this->fExpr,1);
     }
            // Binary operator

    elseif ($this->isBynaryOperator($tok["item"]))
     {
      $tok["type"] = "operator";
      $this->fExpr = substr($this->fExpr,1);
     }
            // Unary operator

    elseif ($this->isUnaryOperator($tok["item"]))
     {
      $tok["UnaryOp"] = $tok["item"];
      $this->fExpr = substr($this->fExpr,1);

      if ($this->getNextToken($localTok,$errMess))
       {
        if ($localTok["type"] == "operand")
         {
          $tok["item"] = /*$tok["UnaryOp"].*/$localTok["item"];
         }
        else
         {

          $errMess = $errMessPrefix.$RevPolNotationConst['OperandShouldFollowUnaryOpertorErrMsg'];
          $this->log->error($errMess);

         }
       }
      else
       {
        $errMess = $errMessPrefix.$RevPolNotationConst['ExpressionUnexpectedEndErrMsg'];
        $this->log->error($errMess);

       }
     }
            // All the rest

    else
     {
      $tok["type"] = "operand";

      if ($this->isQuote($tok["item"]))
       {
        $this->fExpr = substr($this->fExpr,1);

        if (preg_match("/^(.*?)\"/",$this->fExpr,$arr))
         {
          $tok["item"] = $arr[1];
          $this->fExpr = preg_replace("/^(.*?)\"/","",$this->fExpr);
          $this->exprLen = strlen($this->fExpr);
         }
        else
         {
          $errMess = $errMessPrefix.$RevPolNotationConst['ClosingQuoteNotFoundErrMsg'];
          $this->log->error($errMess);
         }
       }
      else
       {
        $localTok = $tok["item"];
        $tok["item"] = "";

        while (($this->exprLen > 0) &&
               (! $this->isDelimiter($localTok)))
         {
          $tok["item"] .= $localTok;
          $this->fExpr = substr($this->fExpr,1);
          $this->exprLen--;
          $localTok = substr($this->fExpr,0,1);
         }
       }
     }

    if ($tok["item"] != "") { return(true);  }
    else                    { return(false); }
   }


  function    isOperator         ($char)
   {
    if ($this->isUnaryOperator ($char) ||
        $this->isBynaryOperator($char) ||
        $this->isBracket       ($char))
     {
      return(true);
     }

    return(false);
   }


  function    isUnaryOperator    ($char)
   {
    if (preg_match("/[\-]/",$char))
     {
      return(true);
     }

    return(false);
   }


  function    isBynaryOperator   ($char)
   {
    if (preg_match("/[\*\+]/",$char))
     {
      return(true);
     }

    return(false);
   }


  function    isDelimiter        ($char)
   {
    if (/*($this->isOperator($char)) ||*/
        ($this->isSpaceChar($char)))
     {
      return(true);
     }

    return(false);
   }


  function    isSpaceChar        ($char)
   {
    if (preg_match("/\s/",$char))
     {
      return(true);
     }

    return(false);
   }


  function    isBracket          ($char)
   {
    if ($this->isOpenBracket ($char) ||
        $this->isCloseBracket($char))
     {
      return(true);
     }

    return(false);
   }


  function    isOpenBracket      ($char)
   {
    if (preg_match("/[(]/",$char))
     {
      return(true);
     }

    return(false);
   }


  function    isQuote            ($char)
   {
    if (preg_match("|\"|",$char))
     {
      return(true);
     }

    return(false);
   }


  function    isCloseBracket     ($char)
   {
    if (preg_match("/[)]/",$char))
     {
      return(true);
     }

    return(false);
   }

  function    debug_array        ($arr)
   {
    while (list(,$val) = each($arr))
     {

     }
   }
              
               /*
                * Checks the order of token types of expression, if the order is
                * incorrect execurion is terminated and problem indication
                * message issued
                */
              
  function    checkValidItemOrder($itemType,
                                  $prevItemType)
   {
    global         $RevPolNotationConst;

    $errMess = "";
    $errMessPrefix = $RevPolNotationConst['ExpressionErrorErrPrefixMsg'];

    switch ($itemType)
     {
      case ("closeBracket") :
       {
        if ($prevItemType == "openBracket") {$errMess = $errMessPrefix.$RevPolNotationConst['BracketsContainEmptyClauseErrMsg'];
                                             $this->log->error($errMess); }
        if ($prevItemType == "operator")    {$errMess = $errMessPrefix.$RevPolNotationConst['OperatorPreceedsClosingBracketErrMsg'];
                                             $this->log->error($errMess); }
       } break;

      case ("openBracket")  :
       {
        if ($prevItemType == "closeBracket") {$errMess = $errMessPrefix.$RevPolNotationConst['BracketsContainEmptyClauseErrMsg'];
                                             $this->log->error($errMess); }
        if ($prevItemType == "operand")      { //die ("Error on expression parsing: между операндом и открывающей скобкой должен идти оператор");
                                             $errMess = $errMessPrefix.$RevPolNotationConst['OperatorLocationNotRightErrMsg'];
                                             $this->log->error($errMess); }
       } break;

      case ("operator")     :
       {
        if ($prevItemType == "operator")    {$errMess = $errMessPrefix.$RevPolNotationConst['OperandExpectedAfterOperatorErrMsg'];
                                             $this->log->error($errMess); }
        if ($prevItemType == "openBracket") {$errMess = $errMessPrefix.$RevPolNotationConst['OperatorCannotFollowOpenningBracketErrMsg'];
                                             $this->log->error($errMess); }
       } break;

      case ("operand")      :
       {
        if ($prevItemType == "operand")      {$errMess = $errMessPrefix.$RevPolNotationConst['OperandCannotFollowOperandErrMsg'];
                                             $this->log->error($errMess); }
        if ($prevItemType == "closeBracket") {$errMess = $errMessPrefix.$RevPolNotationConst['OperandCannotFollowClosingBracketErrMsg'];
                                             $this->log->error($errMess); }
       } break;

      default:
       {

        $errMess = $errMessPrefix.$RevPolNotationConst['TokenTypeUndefinedErrMsg'];
        $this->log->fatal($errMess);
       }
     }

    return($errMess);
   }

  var    $fExpr;
  var    $fOpMap;
  var    $fExprMap;
  var    $fQuteBeginFlag;
 }

?>