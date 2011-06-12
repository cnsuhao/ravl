<?php
/*      

        © BY http://phpMySearch.web4.hm

        */


class    Config
 {
          
           /*
            * Constructor, should be called when class is created
            * could use outer source passed thru $extSource
            */
          


  function    Config             ($extSource = "")
   {
    $this->setErrMess("");
    $this->selfLoad($extSource);
   }
              // Get settings by key

  function    param              ($paramName = "")
   {
    return($this->hash[$paramName]);
   }

             // Set property Key=value

  function    setParam           ($key,
                                  $value)
   {
    $type = gettype($this->hash[$key]);

    switch ($type)
     {
      case ("boolean") :
       {
        $this->hash[$key] = (boolean)$value;
       }
      default :
       {
        $this->hash[$key] = $value;
       }
     }
   }

              // This method should be reliazed in deriving classes

  function    selfLoad           ($extSource = "")
   {
   }

              
               /*
                * This method should be reliazed in deriving classes
                * Saves sources in outer source
                */
              


  function    selfSave           ($extSource = "")
   {
   }
              // return all names of parameters

  function    paramNames         ()
   {
    return(array_keys($this->hash));
   }

              
               /*
                * Returns message about an error, if such occured or
                * blank string
                */

  function    errMess            ()
   {
    $errMess = $this->errMess;
    return($this->errMess);
   }

  function    setErrMess         ($errMess)
   {
    $this->errMess = $errMess;
   }

  // === private: ===
  var $hash = array();
  var $errMess;
 }

?>