// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_STRSTREAM_HEADER
#define RAVL_STRSTREAM_HEADER 1
//////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/Core/Base/StrStream.hh"
//! lib=RavlCore
//! author="Charles Galambos"
//! date="25/02/1999"
//! docentry="Ravl.Core.IO.Streams"
//! userlevel=Default

#include "Ravl/Stream.hh"
#include "Ravl/String.hh"

#if RAVL_HAVE_STREAMASCLASS
#if RAVL_HAVE_STDNAMESPACE
namespace std {
  class ostrstream;
  class istrstream;
}
#else
class ostrstream;
class istrstream;
#endif
#else

#if RAVL_HAVE_ANSICPPHEADERS
#if RAVL_HAVE_STRINGSTREAM
#include <sstream>
#else
#include <strstream>
#endif
#else
#ifndef VISUAL_CPP
#include <strstream.h>
#else
#include <strstrea.h>
#endif
#endif

#endif

namespace RavlN {
  
  ////////////////////////////
  //! userlevel=Normal
  //: Output stream to memory.
  // Wraps the standard library ostrstream class.
  
  class StrOStreamC 
    : public OStreamC
  {
  public:
    StrOStreamC();
    //: Default constructor.
    
    StringC String();
    //: Get text written to stream.
    // Note, this will reset the buffer. 
    
    UIntT Size() const;
    //: Get the number of bytes written so far.
  protected:

#if RAVL_HAVE_STRINGSTREAM
    ostringstream *oss; // Output string stream.
#else
    ostrstream *oss; // Output string stream.
#endif
  };
  
  ////////////////////////////
  //! userlevel=Normal
  //: Input stream from memory.
// Wraps the standard library istrstream class.
  
  class StrIStreamC 
    : public IStreamC
  {
  public:
    StrIStreamC()
      : iss(0)
      {}
    //: Default constructor
    
    StrIStreamC(const StringC &dat);
    //: Constructor.
    
    inline StringC &String() { return buff; }
    //: Get string used as buffer.
    
  protected:
#if RAVL_HAVE_STRINGSTREAM
    istringstream *iss; // Output string stream.
#else
    istrstream *iss; // Output string stream.
#endif
    StringC buff;
  };
}




#endif
