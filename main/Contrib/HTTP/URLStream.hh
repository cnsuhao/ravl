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
#include "Ravl/OS/Filename.hh"

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
    
    ~URLIStreamC();
    //: Destructor

    IntT Error() const {return m_iError;}
    //: Get CURL error code

    StringC ErrorString() const;
    //: Get a string describing the error

  protected:

    FilenameC m_strTemp;
    
    IntT m_iError;

  };
  
}


#endif
