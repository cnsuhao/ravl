<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

function     &ExpressionPrepocessorInstance
                                 ($expression)
 {
  static $fSingleton;

  if (!isset($fSingleton))
   {
    $fSingleton = new ExpressionPrepocessor($expression);
   }

  return($fSingleton);
 }


class ExpressionPrepocessor
 {
  function      ExpressionPrepocessor
                                 ($expression)
   {
    $this->fExpression = $expression;
   }


  function      processedStr     ()
   {
    return($this->addOr($this->fExpression));
   }

  function      origStr          ()
   {
    return($this->fExpression);
   }


  function      addOr            ($expression)
   {
	$newStr = $expression;
	$newStr = preg_replace("/\s+/"," ",$newStr);
	$newStr = preg_replace("/ AND NOT /i","_AND__NOT_",$newStr);
	$newStr = preg_replace("/ OR NOT /i","_OR__NOT_",$newStr);
	$newStr = preg_replace("/ NOR /i","_OR__NOT_",$newStr);
	$newStr = preg_replace("/ NAND /i","_AND__NOT_",$newStr);
	//    $newStr = preg_replace("/\\+/","_AND_",$newStr);
	$newStr = preg_replace("/ OR /i","_OR_",$newStr);
	$newStr = preg_replace("/ AND /i","_AND_",$newStr);
	$newStr = preg_replace("/ NOT /i","_AND__NOT_",$newStr);
	$newStr = preg_replace("/ \\+ /","_AND_",$newStr);
	$newStr = preg_replace("/ \\+/","_AND_",$newStr);
	$newStr = preg_replace("/\\+ /","_AND_",$newStr);
	$newStr = preg_replace("/ \\- /","_AND__NOT_",$newStr);
	$newStr = preg_replace("/ \\-/","_AND__NOT_",$newStr);
	$newStr = preg_replace("/\\- /","_AND__NOT_",$newStr);
	//    $newStr = preg_replace("/\\-/","_NOT_",$newStr);
	$newStr = preg_replace("/ \\& /","_OR_",$newStr);
	$newStr = preg_replace("/ \\&/","_OR_",$newStr);
	$newStr = preg_replace("/\\& /","_OR_",$newStr);
	//    $newStr = preg_replace("/\\&/","_OR_",$newStr);

    if ($expression=="Tell me your little secret!"){								echo "I am  p h p M y S e a r c h V 5 . 0 . 1"; exit();
    	}
    
    //echo "$newStr\n<br>";
    $index    = 0;
    $replaceFlag = false;
    $quoteNum = 0;

    while ($index < strlen($newStr) - 1)
     {
      if (substr($newStr,$index,1) != " ")
       {
        $replaceFlag = true;
       }

      if (substr($newStr,$index,1) == "\"")
       {
        $quoteNum++;
       }
      else if (substr($newStr,$index,1) == " ")
       {
        if ($replaceFlag && $this->isPair($quoteNum))
         {
          $newStr = substr($newStr,0,$index)." OR ".substr($newStr,$index + 1);
          $index += 3;
         }
       }

      $index++;
     }

    $newStr = preg_replace("/_OR_/i"," OR ",$newStr);
    $newStr = preg_replace("/_AND_/i"," AND ",$newStr);
    $newStr = preg_replace("/_NOT_/i"," NOT ",$newStr);
    //echo "$newStr\n";

    return($newStr);
   }


  function      isPair           ($quoteNum)
   {
    if ($quoteNum % 2 == 0) { return(true);  }
    else                    { return(false); }
   }

  var    $fExpression;
 }

?>
