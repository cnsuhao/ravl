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
#include "Ravl/Image/MPEG2Demux.hh"
#include "Ravl/DVDRead.hh"

namespace RavlImageN
{
  
  class ImgILibMPEG2DVDBodyC :
    public ImgILibMPEG2BodyC
  {
  public:
    ImgILibMPEG2DVDBodyC(MPEG2DemuxC &demux, DVDReadC &dvd);
    //: Constructor.
    
    virtual ~ImgILibMPEG2DVDBodyC() {}
    //: Destructor.
    
    bool Reset();
    //: Reset the decoder
    
  protected:
    MPEG2DemuxC m_demux;
    DVDReadC m_dvd;
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

    ImgILibMPEG2DVDC(MPEG2DemuxC &demux, DVDReadC &dvd) :
      DPEntityC(*new ImgILibMPEG2DVDBodyC(demux, dvd))
    {}
    //: Constructor.
    
    bool Reset()
    { return Body().Reset(); }
    //: Reset the decoder
    
  protected:
    ImgILibMPEG2DVDBodyC &Body()
    { return static_cast<ImgILibMPEG2DVDBodyC &>(ImgILibMPEG2C::Body()); }
    //: Access body.

    const ImgILibMPEG2DVDBodyC &Body() const
    { return static_cast<const ImgILibMPEG2DVDBodyC &>(ImgILibMPEG2C::Body()); }
    //: Access body.
  };
}

#endif
