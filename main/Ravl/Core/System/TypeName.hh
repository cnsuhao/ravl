// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLTYPENAME_HEADER
#define RAVLTYPENAME_HEADER 1
///////////////////////////////////////////////
//! userlevel=Advanced
//! docentry="Ravl.Core.IO"
//! rcsid="$Id$"
//! file="Ravl/Core/System/TypeName.hh"
//! lib=RavlCore
//! author="Charles Galambos"
//! date="23/10/98"

#include "Ravl/config.h"

#if RAVL_HAVE_ANSICPPHEADERS
#include <typeinfo>
#else
#include <typeinfo.h>
#endif

namespace RavlN {
  // Unfortunalty the C++ standard does not specify what
  // string is returned from the type_info.name() function.
  // This means if you wish to use files containing with
  // code generated by different compilers you'll get 
  // inconsistant results.
  // These functions are here to help get around the problem.
  
  const char *TypeName(const type_info &info);
  // This funtion will return a standard name
  // for a class if one is known. The function will attempt
  // to generate a standard name if one is not defined. 
  // If this cannot be done a warning is printed on standard
  // error channel and the program will continue with the
  // compiler generated name.
  
  const char *TypeName(const char *name);
  
  // Same as above but uses the type_info.name() directly.
  
  const char *RTypeName(const char *name);
  
  // Reverse lookup.  Given the stanard name find the 
  // compiler specific one. (NOT IMPLEMENTED)
  
  const type_info &RTypeInfo(const char *name);
  
  // Reverse lookup.  Given the stanard name find the 
  // type info for that class.
  
  void AddTypeName(const type_info &info,const char *newname);
  // Set the stanard name to be used for a particular type.
  
  void AddTypeName(const char *sysname,const char *newname);
  // Set the stanard name to be used for a particular type.

  //: Register typename.
  // Class to make it easy to register typename. Use as global
  // variables. <p>
  // e.g. for class xyzC declare the following as global
  // in a .cc file preferably with the definition of xyzC
  //  static TypeNameC typeNamexyzC(typeinfo(xyzC),"xyzC");
  
  class TypeNameC {
  public:
    TypeNameC(const type_info &info,const char *newname) 
      { AddTypeName(info,newname); }
    //: Constructor.
  };
  
}
#endif
