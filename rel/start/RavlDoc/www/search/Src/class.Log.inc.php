<?php
/*      

        й BY http://phpMySearch.web4.hm

        */

/* GutLib::Log - logging module
 *
 *  Designed for convenient work with log-file and can work with one log-file
 *   (in this case it much simpler to work with it) or with unlimited number of
 *  log-files (in this case you will have to create log classes for each
 *  log-file by means of 'new clause' and deprovise it by invoking class method
 *  'done')
 *
 *  One log-file example
 *
 *
 *  Custom setting for our log-file
 *  LogSetGlobalConfig
 *                  (("-fileName"   => 'mymess.log', // name of log-file
 *                    "-autoMargin" => 'true',       // automatically print in
 *                                                      header and footer lines.
 *                    "-startStrFormat" => '[%d %t] Start engine script',
 *                    "-endStrFormat"   => '[%d %t] End engine script'));
 *
 *   $log = &LogInstance();
 *   $log->notice("search engine init Ok");
 *   $log->notice("document 5 deleted by user");
 *   $log->error("document file open error");
 *
 *
 * File mymess.log after the script work outпосле will look like
 *
 * [05.03.2001 09:07:39] Start search script
 * [05.03.2001 09:07:39] [notice] search engine init Ok
 * [05.03.2001 09:07:39] [notice] document 5 deleted by user
 * [05.03.2001 09:07:39] [error] document file open error
 * [05.03.2001 09:07:39] End search script
 */

$cDebugMode = 0;   //  1 - Enable debug mode,      0 - disable

//  The below flags should be defined by LogSetGlobalConfig()

$cDefaultLogFileName = "mess.log";

$cDateFormat = "%d.%m.%y";
$cTimeFormat = "%h:%m:%s";

$cStrFormat       = "[%d %t] [%m] %s";
$cStartStrFormat  = "=> [%d %t] %s";
$cEndStrFormat    = "<= [%d %t] %s";

$cAutoMargin      = 'false'; 
                                /*
                                 * Don't print headers automatically
                                 * This flag points out whether call
                                 * $log->start() automatically after log-file is opened
                                 * and $log->end() before it is closed
                                 */
                             
$cOutFlag         = 'true'; 
                               /*
                                * this flag points out whether to make real
                                * output or not, by default script will print in
                                * normal mood. If this flag is assigned to false
                                * there will be no output to log-file (this
                                * feature is sometimes useful if you want to
                                * disable logging
                                */
                            


   //                 MESSAGE TYPES
   
          /*
           * When you don't need to semaphore what type of error happend use
           * undefined message type
           */

$cNoneType               = 0;

//   simple message (notice)

$cNoticeType             = 1;

//   Warning

$cWarningType            = 2;

//   Error

$cErrorType              = 3;

// Fatal error after which the program can not continue to work

$cFatalErrorType         = 4;


/* Maximum value of predefined message types codes is used for
 * addition your own modes by invoking static method editMesType()
 *
 * For example you would like to add 2 new types of messages
 *   'special notice' and 'internal error'
 *
 *  // Let's define message type codes constants
 *  $cSpecialNoticeMode = $cMaxPreDefinedTypesCode + 1;
 *  $cInternalErrorMode = $cMaxPreDefinedTypesCode + 2;
 *
 *  // Add these types to the logging system
 *  LogEditMesType($cSpecialNoticeMode,'special notice');
 *  LogEditMesType($cInternalErrorMode,'internal error');
 *
 *      // Now you can print to the log these special message types
 *      // In current PHP version you need to create variable of log
 *      // class by invoking function LogInstance()
 *
 *  $log = &LogInstance();
 *  $log->write('Simple mode test of special notice',$cSpecialNoticeMode);
 *
 *  // Warning!!! You can not call it as below
 *  // &LogInstance()->write('Simple mode test of special notice',$cSpecialNoticeMode);
 *  // &LogInstance()->write('Simple mode test of internal error',$cInternalErrorMode);
 *   Probably in new PHP versions it will be possible :(
 *
 */

$cMaxPreDefinedTypesCode = 10;

// predefined message types that can be printed to log-file

$cTypeMessages = array($cNoneType       => 'undef',
                       $cNoticeType     => 'notice',
                       $cWarningType    => 'warning',
                       $cErrorType      => 'error',
                       $cFatalErrorType => 'FATAL');

/* === private :: ::
 * Access to this variable should be performed through function
 * LogInstance()
 *
 * Example LogInstance()->write()
 * $_instance = array();
 *
 * There are some trics to achieve OOP compatibility. We had to make
 *  a static local variable in this function which returns reference
 *  for this variable (_instanceArray())
 */

              /* === public :: ::

               * Function is used to get a copy of object without creation of
               * local object. It is often with log object without creation
               * the copy of object and calling neither constructor nor
               * destructor (they will be invoked automatically)
               * This function is designed exactly for this purpose. We could
               * recommend to  work using this method without creation of the
               * object with constructor 'new'.

               * Example:
               *  $log = &LogInstance();
               *  $log->notice('Notice message');
               * straight call will look like &LogInstance()->notice('Notice message')

               * In PHP you have to call LogSetGlobalConfig() to define log
               * settings, you have to call it before getting the copy of the
               * class through LogInstance()

               * if you have to work with more than one log-files you can get
               * copy of the class by log-file name
               *  $newLog = &LogInstance("test.log");
               *  $newLog->notice("test");
*/



function     &LogInstance        ($logFileName = "")
 {
  global           $cDefaultLogFileName;
  static           $registerShudownFunctionFlag = false;

  $_instance = &_instanceArray();

  if (!$registerShudownFunctionFlag)
   {
    $registerShudownFunctionFlag = true;

    //Registration of devitation function LogEnd (it calls all destructors)

    register_shutdown_function("LogEnd");
   }

  if ($logFileName == "") { $logFileName = $cDefaultLogFileName; }

  if (!isset($_instance[$logFileName]))
   {
    $inst = new Log(array("-fileName" => $logFileName));
    $_instance[$logFileName] = &$inst;
   }

  return($_instance[$logFileName]);
 }

          
             /* Function is used for changing log settings. Gets a hash of
              *  settings

              * "-fileName" -log-file name,
              *
              *                         default value is $cDefaultLogFileName

              * "-autoMargin" - automatically prints header and footer of
              *                 the log
              *   'true'            - print
              *   'false' или undef - don't print
              *                       default value is $cAutoMargin

              * "-outFlag"    - log output semaphore
              *   'true'      - log output is in normal method
              *                 (it is the default mode it would be strange
              *                  if it wouldn't default;)) )
              *   'false' или undef - don't print to the log-file,
              *

              * "-strFormat"   - format string (default value $cStrFormat)
              *   %s - message string
              *   %d - date
              *   %t - time
              *   %m - mode, message type
              *        (simple notice (notice),
              *         warning(warning),
              *         error(error),
              *         fatal error (fatal), after which program cannot run

              * "-timeFormat" - time format string (default value $cTimeFormat)
              *   %h - hours
              *   %m - minutes
              *   %s - seconds

              * "-dateFormat" - date format string (default value $cDateFormat)
              *   %d - day
              *   %m - month
              *   %y - year

              * "-startStrFormat" - start method output format string:
              *                     start("str")
              *                     default value is $cStartStrFormat
              *   %s - message string
              *   %d - date
              *   %t - time

              * "-endStrFormat" - final method output format string: end("str")
              *                   default value $cEndStrFormat
              *   %s - message string
              *   %d - date
              *   %t - time

              * if copy of Log class was created befor invoking this function
              * it will continue to work with old formats. Changing global
              * formats did not influence it. Global formats will be used when
              * new class Log will be created
              */
          


function      LogSetGlobalConfig    ($hesh)
 {
  global           $cDateFormat;
  global           $cTimeFormat;
  global           $cStrFormat;
  global           $cStartStrFormat;
  global           $cEndStrFormat;
  global           $cAutoMargin;
  global           $cDefaultLogFileName;
  global           $cOutFlag;

  if (isset($hesh["-dateFormat"]))     { $cDateFormat         = $hesh["-dateFormat"];     }
  if (isset($hesh["-timeFormat"]))     { $cTimeFormat         = $hesh["-timeFormat"];     }

  if (isset($hesh["-strFormat"]))      { $cStrFormat          = $hesh["-strFormat"];      }
  if (isset($hesh["-startStrFormat"])) { $cStartStrFormat     = $hesh["-startStrFormat"]; }
  if (isset($hesh["-endStrFormat"]))   { $cEndStrFormat       = $hesh["-endStrFormat"];   }

  if (isset($hesh["-autoMargin"]))     { $cAutoMargin         = $hesh["-autoMargin"];     }
  if (isset($hesh["-fileName"]))       { $cDefaultLogFileName = $hesh["-fileName"];       }

  if (isset($hesh["-outFlag"]))        { $cOutFlag            = $hesh["-outFlag"];        }
 }
           
              /* Using the below method you can add your own or alter already
               * existing message types.
               * Examples of defining message types are described before the
               * definition of constant $cMaxPreDefinedTypesCode

               * Example of message type altering (is used for changing
               * qualifier string of message types. For example if you need
               * to change string 'notice', which is default value of
               * $cNoticeType with string '-= notice =-')
               *  LogEditMesType($cNoticeType,'-= notice =-');
               */
          
function      LogEditMesType     ($typeCode,
                                  $typeStr)
 {
  global           $cTypeMessages;

  $cTypeMessages[$typeCode] = $typeStr;
 }

         //   Class which works with log-files

class    Log
 {
  var              $Handle;
  var              $fileName;

  var              $timeFormat;
  var              $dateFormat;
  var              $strFormat;
  var              $startStrFormat;
  var              $endStrFormat;

  var              $autoMargin;
  var              $outFlag;

         
               /*
                * Prints the string to log (you have specify the string to
                * print to log-file and message type as parameters). Message
                * types are defined above. If you don't specify message type
                * it will use the default which is $cNoneType.

                * You can add your own message type or edit existing through
                * function LogEditMesType()
                */

  function    write              ($str = "",
                                  $mode = "")
   {
    global         $cNoneType;

    if ($mode == "") { $mode = $cNoneType;}

    $this->_write($str,$this->strFormat,$mode);
   }
         
               /*
                * Notice message type
                * you need specify only message string as parameter
                */
         
  function    notice             ($str = "")
   {
    global         $cNoticeType;

    $this->write($str,$cNoticeType);
   }

               /*
                * Warning message type
                * you need specify only message string as parameter
                */

  function    warning            ($str = "")
   {
    global         $cWarningType;

    $this->write($str,$cWarningType);
   }
         
               /*
                * Error message type
                * you need specify only message string as parameter
                */

  function    error              ($str = "")
   {
    global         $cErrorType;

    $this->write($str,$cErrorType);
   }

         //    print message about fatal error and finish the program

  function    fatal              ($str = "")
   {
    global         $cFatalErrorType;

    $this->write($str,$cFatalErrorType);
    $this->done();
    die("FATAL error, exam to log-file: $this->fileName for details\n");


    echo "exit()\n";

   }
         
               /*
                * Prints the header after log script starts
                * if -autoMargin = true called automatically
                */
         
  function    start              ($str = "")
   {
    $this->_write($str,$this->startStrFormat);
   }
         
               /*
                * Prints the footer after log script finished
                * if -autoMargin = true called automatically
                */

  function    end                ($str = "")
   {
    $this->_write($str,$this->endStrFormat);
   }

  function    selfClear          ()
   {
    if (isset($this->Handle)) { fclose($this->Handle);
                                unset($this->Handle);  }

    $fp = fopen($this->fileName,"w");
    fclose($fp);
   }

               /*
                * Creates a copy of the class (usually constructor is not
                * called directly and all work with class is handled by
                * getting class copy pointer through LogInstance())
                * You can pass a hash of non-obligatory parameters

                * also read about parameters at LogSetGlobalFormat() description
                */

  function    Log                ($hesh = array())
   {
    $this->init($hesh);
   }

         
              /*
               * Destructor. If you don't work through LogInstance() you have
               * to call this method  at the end of the work otherwise log files
               * could not to be closed correctly

               * If you work through LogInstance() destructor done() will be
               * called automatically when module will be unloaded. At least PHP
               * should handle it in this manner ;)
               */

  function    done               ()
   {
    $this->close();
   }



  // === private:

  function    _write             ($str,
                                  $format,
                                  $mode = "")
   {
    global         $cNoneType;

    if ($mode == "") { $mode = $cNoneType; }

    if ($this->outFlag)
     {
      $this->open();

      if (isset($this->Handle))
       {

        fwrite($this->Handle,$this->prepareStr($str,$format,$mode)) or die("I/O error: write to file $this->fileName");
        fflush($this->Handle);

       }
     }
   }


  function    init               ($hesh)
   {
    global         $cDefaultLogFileName;
    global         $cTimeFormat;
    global         $cDateFormat;
    global         $cStrFormat;
    global         $cStartStrFormat;
    global         $cEndStrFormat;
    global         $cAutoMargin;
    global         $cOutFlag;

    unset($this->Handle);

    $this->fileName       = isset($hesh["-fileName"])       ? $hesh["-fileName"]
                                                            : $cDefaultLogFileName;
    $this->timeFormat     = isset($hesh["-timeFormat"])     ? $hesh["-timeFormat"]
                                                            : $cTimeFormat;
    $this->dateFormat     = isset($hesh["-dateFormat"])     ? $hesh["-dateFormat"]
                                                            : $cDateFormat;

    $this->strFormat      = isset($hesh["-strFormat"])      ? $hesh["-strFormat"]
                                                            : $cStrFormat;
    $this->startStrFormat = isset($hesh["-startStrFormat"]) ? $hesh["-startStrFormat"]
                                                            : $cStartStrFormat;
    $this->endStrFormat   = isset($hesh["-endStrFormat"])   ? $hesh["-endStrFormat"]
                                                            : $cEndStrFormat;

    $this->autoMargin     = isset($hesh["-autoMargin"])     ? $hesh["-autoMargin"]
                                                            : $cAutoMargin;

    $this->outFlag        = isset($hesh["-outFlag"])        ? $hesh["-outFlag"]
                                                            : $cOutFlag;

    $this->autoMargin     = (strtoupper($this->autoMargin) == 'TRUE') ? 1 : 0;
    $this->outFlag        = (strtoupper($this->outFlag)    == 'TRUE') ? 1 : 0;
   }


  function    open               ()
   {
    global         $cDebugMode;

    if (!isset($this->Handle))
     {
      $this->Handle = fopen("$this->fileName","a") or die ("I/O error: open file $this->fileName");
      if ($this->autoMargin) { $this->start(); }
      if ($cDebugMode)       { echo "-> open log $this->fileName and handle = $this->Handle\n"; }
     }
   }


  function    close              ()
   {
    global         $cDebugMode;

    if (isset($this->Handle))
     {
      if ($this->autoMargin) { $this->end(); }
      fclose($this->Handle) or die ("I/O error: close file $this->fileName");
      if ($cDebugMode)       { echo "<- close log $this->fileName\n"; }
	//privat      chmod($this->fileName,0666);
      unset($this->Handle);
     }
   }


  function    prepareStr         ($str,
                                  $format,
                                  $mode)
   {
    global         $cTypeMessages;

    if (!isset($format)) { $format = $str;       }
    if (!isset($mode))   { $mode   = $cNoneType; }

    $localTime = localtime();

    $sec   = $localTime[0];
    $min   = $localTime[1];
    $hour  = $localTime[2];
    $mday  = $localTime[3];
    $mon   = $localTime[4];
    $year  = $localTime[5];
    $wday  = $localTime[6];
    $yday  = $localTime[7];
    $isdst = $localTime[8];

    $year += 1900;

    $sec  = sprintf("%02d",$sec);
    $min  = sprintf("%02d",$min);
    $hour = sprintf("%02d",$hour);

    $mon  = sprintf("%02d",$mon);
    $mday = sprintf("%02d",$mday);

    $dateStr = $this->dateFormat;
    $dateStr = preg_replace("/%d/",$mday,$dateStr);
    $dateStr = preg_replace("/%m/",$mon ,$dateStr);
    $dateStr = preg_replace("/%y/",$year,$dateStr);

    $timeStr = $this->timeFormat;
    $timeStr = preg_replace("/%h/",$hour,$timeStr);
    $timeStr = preg_replace("/%m/",$min ,$timeStr);
    $timeStr = preg_replace("/%s/",$sec ,$timeStr);

    $str = isset($str) ? $str : "";

    $resStr = $format;
    $resStr = preg_replace("/%s/",$str,    $resStr);
    $resStr = preg_replace("/%d/",$dateStr,$resStr);
    $resStr = preg_replace("/%t/",$timeStr,$resStr);

    isset($cTypeMessages[$mode]) ? $resStr = preg_replace("/%m/",$cTypeMessages[$mode]     ,$resStr)
                                 : $resStr = preg_replace("/%m/",$cTypeMessages[$cNoneType],$resStr);

    return($resStr."\n"); //PP->edited in V4.1
   }

 }


function      LogEnd             ()
 {
  $_instance = &_instanceArray();

  while (list($inst,) = each($_instance))
   {
    if (isset($_instance[$inst]))
     {
      $_instance[$inst]->done();

      unset($_instance[$inst]);
     }
   }
 }


function     &_instanceArray     ()
 {
  static           $_staticInstance = array();

  return($_staticInstance);
 }

?>
