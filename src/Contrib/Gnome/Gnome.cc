// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Omniperception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlGUIGnome

#include "Ravl/GUI/Gnome.hh"

#if !RAVL_OS_WIN32
#include <libgnome/libgnome.h>
#endif

#ifndef DATADIR
#ifndef PROJECT_OUT
#define DATADIR ""
#else
#define DATADIR PROJECT_OUT"/share"
#endif
#endif

#ifndef PREFIX
#ifndef PROJECT_OUT
#define PREFIX ""
#else
#define PREFIX PROJECT_OUT
#endif
#endif

#ifndef SYSCONFDIR
#ifndef PROJECT_OUT
#define SYSCONFDIR ""
#else
#define SYSCONFDIR PROJECT_OUT"/etc"
#endif
#endif

#ifndef LIBDIR
#ifndef PROJECT_OUT
#define LIBDIR ""
#else
#define LIBDIR PROJECT_OUT"/lib"
#endif
#endif

namespace RavlGUIN {
  
  //: Initialise gnome library.
  
  bool GnomeInit(const StringC &appName,const StringC &appVersion,int &nargs,char *args[]) {
#if !RAVL_OS_WIN32    
    gnome_program_init (appName,appVersion,
                        LIBGNOME_MODULE,
                        nargs, args,
                        GNOME_PROGRAM_STANDARD_PROPERTIES
                        ,NULL);
#endif
    return true;
  }
  
}
