// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlHTTPIO
//! file="Ravl/Contrib/HTTP/exHTTP.cc"

#include "Ravl/IO/HTTPStream.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/DP/FileFormatIO.hh"
#include "Ravl/Option.hh"
#include "Ravl/EntryPnt.hh"

using namespace RavlN;

int exHTTP(int nargs,char *args[]) {
   
   // Get command-line options
   OptionC opt(nargs,args);
   StringC url = opt.String("","http://ravl.sourceforge.net/","URL to load");
   opt.Check();

   HTTPIStreamC strm(url);

   Sleep(5);
   
   return 0;

}

RAVL_ENTRY_POINT(exHTTP);
