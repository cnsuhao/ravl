// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
#ifndef RAVL_IMGIOMPEG2DVD_HEADER
#define RAVL_IMGIOMPEG2DVD_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/Image/ImgIOMPEG2.hh"

namespace RavlImageN
{
  
  class ImgILibMPEG2DVDBodyC :
    public ImgILibMPEG2BodyC
  {
  public:
    ImgILibMPEG2DVDBodyC();
    //: Constructor.
    
    ~ImgILibMPEG2DVDBodyC();
    //: Destructor.
  };
  class ImgILibMPEG2DVDC :
    public ImgILibMPEG2C
  {
  public:
    ImgILibMPEG2DVDC() :
      DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    ImgILibMPEG2DVDC(bool) :
      DPEntityC(*new ImgILibMPEG2DVDBodyC())
    {}
    //: Constructor.
  };
}

#endif
