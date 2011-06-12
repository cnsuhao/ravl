<?php
/*      

        © BY http://phpMySearch.web4.hm

        */

function     &ConstInstance      ()
 {
  static           $fSingleton;

  if (!isset($fSingleton))
   {
    $fSingleton = new Constants();
    register_shutdown_function("ConstInstanceDone");
   }

  return($fSingleton);
 }


function      ConstInstanceDone  ()
 {
  $inst = &ConstInstance();
  $inst->done();
 }


class    Constants
 {
  // ==== Methods which returns values of constants  ====


  function    mainTableStruct    ()
   {
    return($this->fMainTableStruct);
   }


  function    structTableStruct  ()
   {
    return($this->fStructTableStruct);
   }


  function    tmpTableStruct     ()
   {
    return($this->fTmpTableStruct);
   }


  function    settingsTableStruct()
   {
    return($this->fSettingsTableStruct);
   }


  function    spiderTableStruct  ()
   {
    return($this->fSpiderTableStruct);
   }


  function    outTmpTableStruct  ()
   {
    return($this->fOutTmpTableStruct);
   }


  function    mainTmpTableName   ()
   {
    return($this->fMainTmpTableName);
   }


  function    outTmpTableName    ()
   {
    return($this->fOutTmpTableName);
   }


  // ====================================================
  function    Constants          ()
   {
    $this->init();
   }

  function    done               ()
   {
   }

  function    init               ()
   {
    $this->fCommonTableStructure =<<<END

	URL 			varchar(255) 	NOT NULL default '',
	pageDate             	DATETIME,
	expiresDate          	DATETIME,
	title                	TINYTEXT,
	description          	TINYTEXT,
	keywords             	VARCHAR(255),
	author               	TINYTEXT,
	replyTo              	VARCHAR(50),
	publisher            	TINYTEXT,
	copyright            	TINYTEXT,
	contentLanguage      	VARCHAR(20),
	pageTopic            	VARCHAR(50),
	pageType             	TINYTEXT,
	abstract             	TINYTEXT,
	classification       	TINYTEXT,
	body_1               	VARCHAR(200),
	body_2               	MEDIUMTEXT,
	expiresFlag 		TINYINT(1)	default '0',
	
	UNIQUE KEY URL (URL),
	KEY expiresFlag(expiresFlag),
END;

    // Main table structure

    $this->fMainTableStruct =<<<END
        $this->fCommonTableStructure
        FULLTEXT KEY body_1(body_1,body_2),
	FULLTEXT KEY title(title,keywords)
END;

    // Temporary tables structure

    $this->fTmpTableStruct =<<<END
        $this->fCommonTableStructure
	RelevanceUrl		FLOAT   DEFAULT 0.0,
	RelevancePUrl		FLOAT   DEFAULT 0.0,
	RelevanceTitle		FLOAT   DEFAULT 0.0,        
	RelevanceBody		FLOAT   DEFAULT 0.0
END;

    // Structure of the table which used for results output

    $this->fOutTmpTableStruct =<<<END

        URL                 	 CHAR(255),
        pageDate            	 DATE,
        expiresDate         	 DATETIME,
        keywords             	VARCHAR(255),
        title               	 TINYTEXT,
        description          	TINYTEXT,
        author               	TINYTEXT,
        replyTo              	VARCHAR(50),
        publisher            	TINYTEXT,
        copyright            	TINYTEXT,
        contentLanguage      	VARCHAR(20),
        pageTopic            	VARCHAR(50),
        pageType             	TINYTEXT,
        abstract             	TINYTEXT,
        classification       	TINYTEXT,
        body_1			VARCHAR(200),
        body_2               	MEDIUMTEXT,
	RelevanceUrl		FLOAT   DEFAULT 0.0,
	RelevancePUrl		FLOAT   DEFAULT 0.0,
	RelevanceTitle		FLOAT   DEFAULT 0.0,        
	RelevanceBody		FLOAT   DEFAULT 0.0,
        UNIQUE KEY          (URL)
END;

    $this->fSettingsTableStruct =<<<END
      recNo                       INT NOT NULL DEFAULT 1,
      DBName                      VARCHAR(64),
      DBUser                      VARCHAR(64),
      DBPassword                  VARCHAR(64),
      DBHost                      VARCHAR(64),

      ProxyActive                 TINYINT,
      ProxyHost                   VARCHAR(64),
      ProxyUser                   VARCHAR(64),

      DBMainTableName             VARCHAR(64),
      DBSettingsTableName         VARCHAR(64),
      DBStructTableName           VARCHAR(64),
      DBSpiderStateTableName      VARCHAR(64),

      adminLogin                  VARCHAR(64),
      adminPassword               VARCHAR(64),

      parsingExtArr               TEXT,

      startURLs                   TEXT,
      urlRetrieveNumber           INT,
      blackList                   TEXT,

      searchDeep                  INT,

      outRefsToPage               INT,
      maxPageRef                  INT,

      searchEngineLogFileName     VARCHAR(255),
      spiderEngineLogFileName     VARCHAR(255),
      adminConfigLogFileName      VARCHAR(255),
      adminSessionLong            INT,

      templatesPath               VARCHAR(255),

      PDFConverterURL             VARCHAR(255),
      PDFConverterVarName         VARCHAR(32),
      PDFConverterVarTransMethod  CHAR(4),

      DOCConverterURL             VARCHAR(255),
      DOCConverterVarName         VARCHAR(32),
      DOCConverterVarTransMethod  CHAR(4),
      
      XLSConverterURL             VARCHAR(255),
      XLSConverterVarName         VARCHAR(32),
      XLSConverterVarTransMethod  CHAR(4),
            
      spiderEngineReparseAll      TINYINT,
      spiderAutoRestart           TINYINT,
      spiderOnlySetUpDomain       TINYINT,

      spiderTimeStart             TIME,
      spiderStartDaysPeriod       INT,
      spiderMaxDescriptionLength  INT,
      spiderMaxAuthorLength       INT,
      spiderMaxKeywordLength      INT,

      phpFullPath                 VARCHAR(255),

      UNIQUE KEY                 (recNo)
END;

    $this->fStructTableStruct =<<<END
	own varchar(255) default NULL,
	child varchar(200) default NULL,
	UNIQUE KEY own (own,child)


END;


    $this->fSpiderTableStruct =<<<END
      recNo                       INT NOT NULL DEFAULT 1,
      dateTime                    DATETIME,
      nowStarted                  TINYINT DEFAULT 0,

      UNIQUE KEY                 (recNo)
END;

    // The name of main temporary table

    $this->fMainTmpTableName  = "MainTmp";
    // The name of temporary table containing output information

    $this->fOutTmpTableName   = "OutTmp";
   }

  var              $fMainTableStruct;
  var              $fTmpTableStruct;
  var              $fOutTmpTableStruct;
  var              $fCommonTableStruct;
  var              $fMainTmpTableName;
  var              $fOutTmpTableName;
  var              $fSettingsTableStruct;
  var              $fStructTableStruct;
  var              $fSpiderTableStruct;
 }

?>
