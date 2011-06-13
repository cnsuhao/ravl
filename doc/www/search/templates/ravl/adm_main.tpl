<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

<html>

	<head>
		<title>phpMySearch - Admintool</title>
		<style>
        .links
        {
                font-weight : bold;
                font-family : Verdana, Geneva, Arial, Helvetica, sans-serif;
                font-size : 11px;
                color : #000000;
        }
        .titext
        {
                font-weight : bold;
                font-family : Verdana, Geneva, Arial, Helvetica, sans-serif;
                font-size : 13px;
                color : #0000bb;
        }
        .normal
        {
                font-weight : bold;
                font-family : Verdana, Geneva, Arial, Helvetica, sans-serif;
                font-size : 11px;
                color : #000000;
        }
        .s10
        {
                font-family : Verdana, Geneva, Arial, Helvetica, sans-serif;
                font-size : 10px;
                color : #000000;
        }
</style>
	</head>

	<body bgcolor="#ffffff">
		<p>&nbsp;<br>
		</p>
		<h1 align="center"><a href="http://phpMySearch.web4.hm" target="_blank" title="phpMySearch - MySQL and PHP search engine"><img src="docs/button2.gif" border="0" alt="phpMySearch - MySQL and PHP search engine"></a><b><br>
				
				phpMySearch V5.0</b></h1>
		<p class="normal"><br>
			<a href="manual/index.html" target="_blank">phpMySearch manual</a><br>
			<br>
			<a href="http://phpMySearch.web4.hm" target="_blank">Official phpMySearch Homepage</a><a href="http://web4.hm/phpMySearch" target="_blank"><br>
				<br>
			</a><a href="docs/phpinfo.php" target="_blank">Show PHP information<br>
			</a><br>
		</p>
		
		{ERROR_MESSAGE} {MESSAGE}
		<form action="admin.php" method="post">
			<div align="left">
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td>
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td colspan="2"><font class="titext">MySQL Database settings:</font> <input type="hidden" name="sid" value="{sid}"> <input type="hidden" name="currPath" value="{currPath}"><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td class="normal" colspan="2"><font color="blue">{PagesInDatabase} Pages in database</font></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Clear DB:</td>
												<td><input type="submit" name="clearTable" value="Clear" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Optimize DB:</td>
												<td><input type="submit" name="optimizeTable" value="Force" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" colspan="2">
													<div align="center">
                        <input type="submit" name="submit" value="Submit" class="normal">
                        <input type="submit" name="logout" value="Logout" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td>
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td valign="top" class="normal" colspan="3"><font class="titext">Spider settings:</font><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td valign="top" class="normal" bgcolor="#dddddd">Search start URL's:</td>
												<td class="s10" colspan="2">{startURLs}</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Number of attempts to retrieve the page:</td>
												<td><input type="text" name="urlRetrieveNumber" value="{urlRetrieveNumber}" size="2" maxlength="2" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" valign="top" bgcolor="#dddddd">Not indexed URL's (Black list):</td>
												<td class="s10" colspan="2">{blackList}</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Delete urls from DB starting at:</td>
												<td><input type="text" name="deletedURL" value="" size="15" maxlength="255" class="normal"><input type="submit" name="URLDelete" value="Delete" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Parse only set domain:</td>
												<td><input type="checkbox" name="spiderOnlySetUpDomain" value="1" class="normal" {spiderOnlySetUpDomain_checked}></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd" valign="top">Document extensions to index:</td>
												<td class="s10">{parsingExtArr}</td>
												<td class="s10"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Document description max length:</td>
												<td><input type="text" name="spiderMaxDescriptionLength" value="{spiderMaxDescriptionLength}" size="12" maxlength="5" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Document author max length:</td>
												<td><input type="text" name="spiderMaxAuthorLength" value="{spiderMaxAuthorLength}" size="12" maxlength="5" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Document keywords max length:</td>
												<td><input type="text" name="spiderMaxKeywordLength" value="{spiderMaxKeywordLength}" size="12" maxlength="5" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Search depth:</td>
												<td><input type="text" name="searchDeep" value="{searchDeep}" size="3" maxlength="3" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Reparse all:</td>
												<td><input type="checkbox" name="spiderEngineReparseAll" value="1" class="normal" {spiderEngineReparseAll_checked}></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Automatic spider start:</td>
												<td><input type="checkbox" name="spiderAutoRestart" value="1" class="normal" {spiderAutoRestart_checked}></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Start time:</td>
												<td><input type="text" name="spiderTimeStart" value="{spiderTimeStart}" size="8" maxlength="8" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Start spider each(days):</td>
												<td><input type="text" name="spiderStartDaysPeriod" value="{spiderStartDaysPeriod}" size="3" maxlength="3" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Force start/stop crawling:</td>
												<td><input type="submit" name="SpiderStart" value="Start spider" class="normal"><input type="submit" name="SpiderStop" value="Stop spider" class="normal">
                      <br>
                      <span class="normal"><a href="spider.php" target="_blank">click 
                      here for the alternative methode</a></span><br>
                    </td>
												<td class="normal"><font color="blue">(status: <b>{spiderState}</b>)</font></td>
											</tr>
											<tr>
												<td class="normal" colspan="3">
													<div align="center">
														<input type="submit" name="submit" value="Submit" class="normal"><input type="reset" name="submit" value="Reset" class="normal"><input type="submit" name="logout" value="Logout" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td>
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td class="normal" colspan="2"><font class="titext">Search result presentation settings:</font><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Number of links per page:</td>
												<td><input type="text" name="outRefsToPage" value="{outRefsToPage}" size="3" maxlength="3" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Max pages block:</td>
												<td><input type="text" name="maxPageRef" value="{maxPageRef}" size="3" maxlength="3" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" colspan="2">
													<div align="center">
														<input type="submit" name="submit" value="Submit" class="normal"><input type="reset" name="submit" value="Reset" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td valign="top">
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td class="normal" colspan="2"><font class="titext">Proxy settings:</font><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Proxy settings active:</td>
												<td><input type="checkbox" name="ProxyActive" value="1" class="normal" {ProxyActive_checked}></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Proxy host:</td>
												<td><input type="text" name="ProxyHost" value="{ProxyHost}" size="40" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Proxy user and pass:</td>
												<td><input type="text" name="ProxyUser" value="{ProxyUser}" size="40" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" colspan="2">
													<div align="center">
														<input type="submit" name="submit" value="Submit" class="normal"> <input type="reset" name="submit" value="Reset" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td>
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td class="normal" colspan="2"><font class="titext">Converter settings:</font><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">PDF converter URL:</td>
												<td><input type="text" name="PDFConverterURL" value="{PDFConverterURL}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">PDF Variablename:</td>
												<td><input type="text" name=" PDFConverterVarName" value="{PDFConverterVarName}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">DOC converter URL:</td>
												<td><input type="text" name="DOCConverterURL" value="{DOCConverterURL}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">DOC Variablename:</td>
												<td><input type="text" name="DOCConverterVarName" value="{DOCConverterVarName}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">XLS converter URL:</td>
												<td><input type="text" name="XLSConverterURL" value="{XLSConverterURL}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">XLS Variablename:</td>
												<td><input type="text" name="XLSConverterVarName" value="{XLSConverterVarName}" size="50" maxlength="255" class="normal"></td>
											</tr>
											<tr>
												<td class="normal" colspan="2">
													<div align="center">
														<input type="submit" name="submit" value="Submit" class="normal"> <input type="reset" name="submit" value="Reset" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<table border="0" cellpadding="0" cellspacing="1" bgcolor="black">
					<tr>
						<td>
							<table border="0" cellpadding="0" cellspacing="1" bgcolor="white">
								<tr>
									<td>
										<table border="0" cellspacing="4" cellpadding="2">
											<tr>
												<td class="normal" colspan="3"><font class="titext">Other settings:</font><br>
													<hr align="left" size="1" noshade>
												</td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">SearchEngine log FileName:</td>
												<td><input type="text" name="searchEngineLogFileName" value="{searchEngineLogFileName}" size="40" maxlength="255" class="normal"></td>
												<td class="normal"><font color="blue">Filesize: {searchEngineLogFileSize}</font></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">SpiderEngine log FileName:</td>
												<td><input type="text" name="spiderEngineLogFileName" value="{spiderEngineLogFileName}" size="40" maxlength="255" class="normal"></td>
												<td class="normal"><font color="blue">Filesize: {spiderEngineLogFileSize}</font></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">AdminTool log FileName:</td>
												<td><input type="text" name="adminConfigLogFileName" value="{adminConfigLogFileName}" size="40" maxlength="255" class="normal"></td>
												<td class="normal"><font color="blue">Filesize: {adminConfigLogFileSize}</font></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Clear log files:</td>
												<td><input type="submit" name="ClearLog" value="Clear" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Templates path:</td>
												<td><input type="text" name="templatesPath" value="{templatesPath}" size="40" maxlength="255" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">PHP FullPath</td>
												<td><input type="text" name="phpFullPath" value="{phpFullPath}" size="40" maxlength="255" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Admin Login:</td>
												<td><input type="text" name="adminLogin" value="{adminLogin}" size="20" maxlength="20" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Admin Password:</td>
												<td><input type="password" name="adminPassword" value="" size="20" maxlength="20" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" bgcolor="#dddddd">Confirm password:</td>
												<td><input type="password" name="confirmPassword" value="" size="20" maxlength="20" class="normal"></td>
												<td></td>
											</tr>
											<tr>
												<td class="normal" colspan="3">
													<div align="center">
														<input type="submit" name="submit" value="Submit" class="normal"> <input type="reset" name="submit" value="Reset" class="normal"> <input type="submit" name="logout" value="Logout" class="normal"></div>
												</td>
											</tr>
										</table>
									</td>
								</tr>
							</table>
						</td>
					</tr>
				</table>
				<br>
				<br>
			</div>
		</form>
	</body>

</html>
