<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

$sessionTableName      = "phpMySearch_Sessions";    
                                          /*
                                           * Name of the session where stored
                                           * session IDs and expiration time
                                           */
                                         


$sessionValueTableName = "phpMySearch_SessionsVal"; 
                                          /*
                                           * Name of the table where session
                                           * variables are stored
                                           */
                                         


// === public:


/*
 * You should work with class in the following way:
 * create an object of SessionManager class
 *
 *   $manager  = new SessionManager("test","10.0.0.1","user","password");
 *
 *      Then you you have to get a copy of Session class either by creating
 *       new session or by getting existing one by its ID
 * $session = $manager->get(1212);          // Get session by ID
 * $session = $manager->create("login",20); // Create new session 20 minute long
 *
 * $session->setValue("varname","value");
 *
 * if ($session->valid()) // check if session exists
 *     {
 *      $value = $session->getValue("varname"); // get the value
 *
 *      $session->setValues(array("varname1" => "value1",
 *                                "varname1" => "value2"));
 *      $hesh = $session->getValues(); // get hash of session values
 *     }
 *
 * $manager->destroy(); // manager deinitialization, you are obligated to call it
 */



class    SessionManager
 {

  function    SessionManager     ($DBName,
                                  $Host,
                                  $User,
                                  $Password)
   {
    $this->init($DBName,$Host,$User,$Password);
   }

              
               /*
                * destroy the manager
                * you are obligated to call it at the end of work with class
                */
              


  function    destroy            ()
   {
    $this->DBDeinit();
   }

              
               /*
                * create session, returns a copy of Session class, thru which
                * will be handled all the work with created session
                *
                * if you need just get a copy of session object you should call
                * get method
                *
                * as parameters you should pass a string (it can be login) and
                * length of the session in minutes
                */
              


  function    create             ($login,
                                  $expired)
   {
    global $sessionTableName;

    $id = generateID($login);

    $query = "REPLACE INTO $sessionTableName (ID, Expired) VALUES('$id',now() + INTERVAL $expired MINUTE)";

    if (!mysql_query($query,$this->link))
     {
      die ();
     }

    return($this->get($id));
   }

              
               /*
                * Get session by indetifier
                * returns a copy of the class Session
                */
              



  function    get                ($id)
   {
    return(new Session($this,$id));
   }



  // === protected:

  
   /*
    ******************************************************************************
    * The following mathods should not be called directly as all work is handled *
    * through copy of Session class nad they will be called from that class      *
    ******************************************************************************
    */
  


              // kill the session

  function    invalidate         ($id)
   {
    global $sessionTableName;
    global $sessionValueTableName;

    $query = "DELETE FROM $sessionTableName WHERE ID = '$id'";
    mysql_query($query,$this->link) or die("Bad query: $query => (".mysql_error($this->link).")");

    $query = "DELETE FROM $sessionValueTableName WHERE ID = '$id'";
    mysql_query($query,$this->link) or die("Bad query: $query => (".mysql_error($this->link).")");
   }

              
               /*
                * You have to pass session ID as parameter
                * Returns :
                *    true  - if such session exists
                *    false - if there is no such session or it is expired
                */
              
          
  function    valid              ($id)
   {
    global $sessionTableName;

    $res = false;

    $this->invalidateExpires();

    $query = "SELECT * FROM $sessionTableName WHERE ID = '$id'";
    $result = mysql_query($query,$this->link) or die("Bad query: $query => (".mysql_error($this->link).")");

    if ($result)
     {
      if (mysql_num_rows($result) > 0)
       {
        $res = true;
       }
     }

    return($res);
   }

              // saves hash of variables to the database

  function    setValues          ($id,
                                  $hesh)
   {
    global $sessionValueTableName;

    if ($this->valid($id))
     {
      while (list($key,$val) = each($hesh))
       {
        $query = "REPLACE INTO $sessionValueTableName (ID,name,value) VALUES('$id','$key','$val')";
        mysql_query($query,$this->link) or die("Bad query: $query => (".mysql_error($this->link).")");
       }
     }
   }

              // get hash of variables for the session


  function    getValues          ($id)
   {
    global $sessionValueTableName;

    $hesh = array();

    if ($this->valid($id))
     {
      $query = "SELECT name,value FROM $sessionValueTableName WHERE ID = '$id'";
      $result = mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");

      while (list($name,$value) = mysql_fetch_row($result))
       {
        $hesh[$name] = $value;
       }
     }

    return($hesh);
   }


  // === private:
              // Kill expired sessions

  function    invalidateExpires  ()
   {
    global $sessionTableName;
    global $sessionValueTableName;

    $query = "SELECT ID FROM $sessionTableName WHERE Expired <= now()";
    $result = mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");

    while (list($id) = mysql_fetch_row($result))
     {
      $query = "DELETE FROM $sessionTableName WHERE ID = '$id'";
      mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");

      $query = "DELETE FROM $sessionValueTableName WHERE ID = '$id'";
      mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");
     }
   }

  function    init               ($DBName,
                                  $Host,
                                  $User,
                                  $Password)
   {
    $this->fDBName       = $DBName;
    $this->fHost         = $Host;
    $this->fUserName     = $User;
    $this->fUserPassword = $Password;

    $this->DBInit();
   }


  function    connect            ()
   {
    $this->link = mysql_connect
                    ($this->fHost,
                     $this->fUserName,
                     $this->fUserPassword);

    if ($this->link > 0)
     {

      mysql_select_db($this->fDBName,$this->link) or die ("Select DB error: $fDBName");
     }
    else
     {
      // trap DataBase connection opening error
      die ("Could not connect to DataBase");
     }
   }


  function    disconnect         ()
   {
    mysql_close($this->link);
   }


  function    DBInit             ()
   {
    global $sessionTableName;
    global $sessionValueTableName;

    $this->connect();

    $query = "CREATE TABLE IF NOT EXISTS $sessionTableName (ID char(20) NOT NULL PRIMARY KEY, Expired DATETIME);";
    mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");

    $query = "CREATE TABLE IF NOT EXISTS $sessionValueTableName (ID char(20) NOT NULL,name varchar(255) NOT NULL,value varchar(255), PRIMARY KEY (ID,name))";
    mysql_query($query,$this->link) or ("Bad query: $query => (".mysql_error($this->link).")");
   }


  function    DBDeinit           ()
   {
    $this->disconnect();
   }

  var    $fDBH;
  var    $fDBName;
  var    $fHost;
  var    $fUserName;
  var    $fUserPassword;
  var    $link;
 }

?>