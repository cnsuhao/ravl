<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

/*
 * Objects of this class are not created directly. You have to create the
 * object by invoking create or get method of SessionManager class.

 * !!! NOTE !!!
 *       After invoking destroy method of SessionManager class the
 *       work thru SessionManager with copies of this class is
 *       unpredictable

 * After creation of the copy of class it is filled with variables of
 * this session. It means that if you have put variable to the session
 * you cannot get the value of the variable until you invoke get method
 * of SessionManager or invoking refresh method of this class.

 */

class Session
 {
  // === public:

              // is this session valid (exists) ?

  function    valid              ()
   {
    return($this->fManager->valid($this->fID));
   }

              // kill the session

  function    invalidate         ()
   {
    $this->fManager->invalidate($this->fID);
    $this->fData = array();
   }

              // get current session ID

  function    getID              ()
   {
    return($this->fID);
   }

              // get hash of session values

  function    getValues          ()
   {
    return($this->fData);
   }

              // get the value by variable's name

  function    getValue           ($varName)
   {
    return($this->fData[$varName]);
   }

              // save hash data to the session

  function    setValues          ($hash)
   {
    while (list($key,$val) = each($hesh))
     {
      $this->setValue($key,$val);
     }
   }

              
               /*
                * Saves variable to the session
                *
                * parameters: variable name,
                *             variable value
                *
                */
              


  function    setValue           ($varName,
                                  $varValue)
   {
    $this->fData[$varName] = $varValue;
    $this->fManager->setValues($this->fID,array($varName => $varValue));
   }


  // === private:
              
               /*
                * Private constructor
                *
                * Parameters :
                * manager - copy of SessionManager class thru which handled all
                *           low level work with session
                *      id - session indentifier
                */
              


  function    Session           (&$manager,
                                  $id)
   {
    $this->init($manager,$id);
   }


  function    init               (&$manager,
                                   $id)
   {
    $this->fData    = array();
    $this->fID      = $id;
    $this->fManager = $manager;

    $this->refresh();
   }

              
               /*
                * refreshes variables for this session
                * called automatically when copy of the class is created
                */
              


  function    refresh            ()
   {
    $this->fData = $this->fManager->getValues($this->fID);
   }

  var    $fManager;
  var    $fData;
  var    $fID;
 }

?>