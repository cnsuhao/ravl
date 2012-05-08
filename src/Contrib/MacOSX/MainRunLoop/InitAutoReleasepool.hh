// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2005, Charles Galambos.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_INITAUTORELEASEPOOL_HEADER
#define RAVL_INITAUTORELEASEPOOL_HEADER 1
//! lib=RavlMacOSXRunLoop

namespace RavlN {

  //! Make sure release pool is initialised for the current thread.
  bool AutoReleasepoolInit();

  //! Call release for the current pool
  bool AutoReleasepoolRelase();

}

#endif
