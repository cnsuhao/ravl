// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_CSPCONTROL_HEADER
#define RAVLIMAGE_CSPCONTROL_HEADER 1
////////////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! author="Charles Galambos"
//! docentry="Ravl.Contrib.Video IO.ClipStationPro"
//! lib=CSPDriver
//! file="Contrib/ClipStationPro/CSPControl.hh"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteYUV422Value.hh"
#include "Ravl/TimeCode.hh"

extern "C" {
#include <dvs_clib.h>
#include <dvs_fifo.h>
}

namespace RavlImageN {
  
  //! userlevel=Develop
  //: Clip station control class.
  
  class ClipStationProDeviceC {
  public:
    ClipStationProDeviceC(const StringC &devName,const ImageRectangleC &nrect);
    //: Constructor.
    
    ~ClipStationProDeviceC();
    //: Destructor.
    
    bool Init();
    //: Setup video modes to a suitable default.
    
    bool GetFrame(void *buff,int x,int y);
    //: Get one frame of video.
    
    BufferC<ByteYUV422ValueC> GetFrame();
    //: Get one field of video.
    
    bool PutFrame(void *buff,int x,int y);
    //: Get one frame of video.
    
    bool CSPGetAttr(const StringC &attrName,StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool CSPSetAttr(const StringC &attrName,const StringC &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

  protected:
    sv_handle *dev;
    sv_fifo *fifo;
    sv_fifo_configinfo fifo_config;
    
    ImageRectangleC rect;
    bool useDMA;  // Use DMA transfer's.
    bool doInput;
    bool fifoMode; // Use fifo ?
    bool captureAudio; // Capture audio ?
    bool captureVideo; // Capture video ?

    int frameBufferSize;

    TimeCodeC timecode_from_getframe;
  };

}

#endif
