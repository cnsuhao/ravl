// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlURLIO
//! file="Ravl/Contrib/HTTP/exURL.cc"

#include "Ravl/IO/URLStream.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/DP/FileFormatIO.hh"
#include "Ravl/Option.hh"
#include "Ravl/EntryPnt.hh"

using namespace RavlN;

int exURL(int nargs,char *args[]) {
   
   // Get command-line options
   OptionC opt(nargs,args);
   StringC url = opt.String("","http://ravl.sourceforge.net","URL to load");
   StringC out = opt.String("","-","Where to write data. ");
   opt.Check();
   
   OStreamC os(out);
   URLIStreamC strm(url);
   
   if (strm.Error()) cerr << strm.ErrorString() << endl;

   strm.CopyTo(os);
   
   return 0;

}

RAVL_ENTRY_POINT(exURL);
