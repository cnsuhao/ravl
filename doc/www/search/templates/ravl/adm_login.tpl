<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

<html>

	<head>
		<title>phpMySearch - Admintool</title>
		<style>
      .normal
        {   font-weight : bold;
            font-family : Verdana, Geneva, Arial, Helvetica, sans-serif;
            font-size : 11px;
            color : #000000;       }
      </style>
	</head>

	<body>
		<div align="center">
			{ERROR_MESSAGE}
			<form action="./admin.php" method="post">
				<div align="center">
					<p><img src="http://phpMySearch.web4.hm/phpMySearch/images/phpMySearch-button.gif" border="0"><br>
						<br>
					</p>
					<table border="0" align="center">
						<tr>
							<td class="normal">Login:</td>
							<td><input type="text" name="name" size="12" maxlength="36" class="normal"></td>
						</tr>
						<tr>
							<td class="normal">Password:</td>
							<td><input type="password" name="pswd" size="12" maxlength="36" class="normal"></td>
						</tr>
						<tr>
							<td align="right" colspan="2">
								<div align="center">
									<input type="submit" name="enter" value="Enter" class="normal"></div>
							</td>
						</tr>
					</table>
				</div>
			</form>
		</div>
	</body>

</html>