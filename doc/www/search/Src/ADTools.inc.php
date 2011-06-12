<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

/* Generator - ID generator module
 *
 * Module is aimed to generate session ID
 *
 */


$MaxIDCipherNumber           = 15;  
                                     /*
                                      * Maximum ID length
                                      * Recommended that  $MaxIDCipherNumber>13
                                      *
                                      */
                                    


$MaxPartAdditionRandValue    = 867; // Constant -> Recomended not to change


$MaxPartAdditionCifherNumber = 2;   // Constant -> Recomended not to change


function      generateID         ($login)
 {
  global      $MaxIDCipherNumber;
  global      $MaxPartAdditionRandValue;
  global      $MaxPartAdditionCifherNumber;

  $begin = "";
  $res   = "";

  srand ((double) microtime() * 1000000);
  $login = md5($login);

  for ($i = 0;$i < strlen($login);$i++)
   {
    $begin .= ord(substr($login,$i,1));
   }

  $begin = substr($begin,0,$MaxIDCipherNumber - 5);

  for ($i = 0;$i < strlen($begin);$i += $MaxPartAdditionCifherNumber)
   {
    $res .= (substr($begin,$i,$MaxPartAdditionCifherNumber) + floor(rand(0,$MaxPartAdditionRandValue)));
   }

  return(strrev(substr($res,0,$MaxIDCipherNumber)));
 }

?>