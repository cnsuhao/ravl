<?php
/*      

        © BY http://phpMySearch.web4.hm

        */
        
//class.RevPolNotation.inc.php
$RevPolNotationConst = array
 (
  'LogTokenInfoMsg'                            => "RevPol: token: %s (%s)",
  'ExpressionErrorErrPrefixMsg'                => "Error in expression: ",
  'ClosingBracketWithoutOpenErrMsg'            => "found a closing bracket without matching opened bracket",
  'TokenTypeUndefinedErrMsg'                   => "undefined token type (fatal error)",
  'OperandShouldFollowUnaryOpertorErrMsg'      => "operand should follow the unary operator",
  'ExpressionUnexpectedEndErrMesg'             => "unexpected end",
  'ClosingQuoteNotFoundErrMsg'                 => "closing quote not found",
  'BracketsContainEmptyClauseErrMsg'           => "brackets contain an empty clause",
  'OperatorPreceedsClosingBracketErrMsg'       => "operator preceeds closing bracket",
  'BracketsContainEmptyClauseErrMsg'           => "brackets contain an empty clause",
  'OperatorLocationNotRightErrMsg'             => "operator should be placed between operand and opening bracket",
  'OperandExpectedAfterOperatorErrMsg'         => "only operand is expected after operator",
  'OperatorCannotFollowOpenningBracketErrMsg'  => "operator cannot follow the openning bracket",
  'OperandCannotFollowOperandErrMsg'           => "operand cannot follow the operand",
  'OperandCannotFollowClosingBracketErrMsg'    => "operand cannot follow the closing bracket"
 );

//class.SearchEngine.inc.php
$SearchEngineConst = array
 (
  'DataBaseErrPrefixMsg'                       => "DataBase error: ",
  'SelectDBErrMsg'                             => "Select DB error: %s",
  'DataBaseConnectionErrMsg'                   => "Could not connect to DataBase: %s",
  'ExpressionParsingErrPrefixMsg'              => "Expression parsing error: ",
  'ExpressionParsingStartMsg'                  => "start expression parsing: %s",
  'TokenInSequenceMsg'                         => "token in sequence: %s",
  'OperandPushedToArrayMsg'                    => "operand: %s pushed to operand array",
  'EmptyOperandStackErrMsg'                    => "operand stack is empty (after operator must be operand)",
  'ExecuteOperationMsg'                        => "operator: %s -> execute operation",
  'OperationInfoMsg'                           => "Executed operation: %s %s %s",
  'OperationResultMsg'                         => "Result => %s",
  'TokenTypeUnknownErrMsg'                     => "Unknown token type '%s'",
  'StackPopResultValueMsg'                     => "Pop result value from stack",
  'ExpressionEmptyErrMsg'                      => "The expression is empty",
  'DBSelectionErrMsg'                          => "DataBase selection error",
  'BadQueryErrMsg'                             => "Bad query: %s => (%s)",
  'UndefinedOperatorErrMsg'                    => "Undefined operator: '%s'",
  'LogQueryInfoMsg'                            => "Query: %s",
  'LogRowCountInfoMsg'                         => "Row count: %d",
  'TmpTableCreationErrorErrMsg'                => "DataBase tmp table creation error",
  'ANDOperationMsg'                            => "Execution of operation 'AND'",
  'OROperationMsg'                             => "Execution of operation 'OR'",
  'GetSelectionExecutedMsg'                    => "get Selection executed",
  'GetNotSelectionExecutedMsg'                 => "get NotSelection executed",
  'PrepareOutResultMsg'                        => "prepare out result",
  'DBExpiredRecodsDeletedMsg'                  => "DataBase delete expired records error",
  'WrongPHPVersion'			       => "You have wrong PHP-version for phpMySearch. Please check manual.",
  'NoCurlDetected'			       => "phpMySearch detected no Curl. Curl is recommend for the spider. Please check manual.",
  'WrongMySQLVersion'			       => "phpMySearch detected wrong MySQL-Version. Please check manual."
 );

$SpidrEngineConst = array
 (
  'SpiderStartingInfoMsg'                      => "Spider starting",
  'SpiderAlreadyStartedInfoMsg'                => "Spider already started",
  'SpiderEndInfoMsg'                           => "Spider stop",
  'DBConnectOkMsg'                             => "dataBase connect ok",
  'DBSelectErrMsg'                             => "Select DB error: %s",
  'DBConnectionErrMsg'                         => "Could not connect to DataBase",
  'NoStartURLsWrnMsg'                          => "no start urls",
  'StartURLInfoMsg'                            => "start: %s",
  'URLInfoMsg'                                 => "URL = (%s)=> %s",
  'PageLoadStartMsg'                           => "start load",
  'PageLoadedMsg'                              => "page loaded",
  'PageRefNotFoundInfo'                        => "Refs in page not found",
  'PageLoadingErrMsg'                          => "Error in page loading by means curlLib => (%s) %s",
  'BadQueryErrMsg'                             => "Bad query: %s => (%s)",
  'PageNeedParseInfoMsg'                       => "need parse",
  'PageNotNeedParseInfoMsg'                    => "not need parse",
  'URLInBlackListInfoMsg'                      => "URL: %s in blacklist",
  'LogQueryInfoMsg'                            => "Query: %s",
  'LogRowCountInfoMsg'                         => "Row count: %d",
  'RetrieveUrlInfoMsg'                         => "One more attempt to retrieve page"
 );

//class.SearchVisualizer.inc.php
$SearchVisualizerConst = array
 (
  'StartSearchEngineMsg'                       => "Search engine start",
  'SearchEngineConstructedMsg'                 => "Search engine constructed",
  'RequestStringInfoMsg'                       => "Requst string = %s",
  'ArrayLengthInfoMsg'                         => "array length = %s",
  'AllPagesNumInfoMsg'                         => "allPagesNum = %s",
  'NeedOutPageInfoMsg'                         => "needOutPage = %s",
  'BeginPageNumInSqMsg'                        => "beginPageNumInSq = %s",
  'EndPageNumInSqInfoMsg'                      => "endPageNumInSq = %s",
  'PrevPagesRefFlagInfoMsg'                    => "prevPagesRefFlag = %s",
  'NextPagesRefFlagInfoMsg'                    => "nextPagesRefFlag = %s",
  'PageInfoMsg'                                => "page   = %s",
  'SearchRequestStrInfoMsg'                    => "search = %s"
 );
?>

