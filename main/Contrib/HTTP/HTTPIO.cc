// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlHTTPIO
//! file="Ravl/Contrib/HTTP/HTTPIO.cc"

namespace RavlN {
  
  extern void InitHTTPStreamIO();
  
  void InitHTTP() {
     InitHTTPStreamIO();
  }

}
