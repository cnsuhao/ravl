// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_URLISTREAM_HEADER
#define RAVL_URLISTREAM_HEADER 1
/////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlURLIO
//! author="James Smith"
//! docentry="Ravl.IO.URLs"
//! file="Ravl/Contrib/HTTP/URLIStream.hh"
//! date="30/10/2002"

#include "Ravl/Stream.hh"

namespace RavlN {

   ////////////////////////////
   //! userlevel=Normal
   //: Get a URL
  
   class URLIStreamC 
      : public IStreamC
   {
   public:

      URLIStreamC()
      {}
      //: Default constructor
    
      URLIStreamC(const StringC &url,bool buffered=true);
      //: Open net connection for input
    
      IntT Push(void *ptr, size_t size, size_t nmemb);
      //: Pushes data onto the stream from curl

   protected:

      bool Get(StringC& url);
      //: Initialises the download of the URL

      int fd[2];

   };
  
}


#endif
