// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_HTTPISTREAM_HEADER
#define RAVL_HTTPISTREAM_HEADER 1
/////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlHTTPIO
//! author="James Smith"
//! docentry="Ravl.IO.HTTP"
//! file="Ravl/Contrib/HTTP/HTTPIStream.hh"
//! date="30/10/2002"

#include "Ravl/Stream.hh"

namespace RavlN {

   ////////////////////////////
   //! userlevel=Normal
   //: Get an HTTP URL
  
   class HTTPIStreamC 
      : public IStreamC
   {
   public:

      HTTPIStreamC()
      {}
      //: Default constructor
    
      HTTPIStreamC(const StringC &url);
      //: Open net connection for input
    
   protected:

      bool Get(StringC& url);
      //: Initialises the download of the URL

   public:

      IntT Push(void *ptr, size_t size, size_t nmemb);
      //: Pushes data onto the stream from curl

   };
  
}


#endif
