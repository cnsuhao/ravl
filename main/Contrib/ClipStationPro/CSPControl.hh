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
//! file="Ravl/Contrib/ClipStationPro/CSPControl.hh"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteYUV422Value.hh"
#include "Ravl/TimeCode.hh"

extern "C" {
#include <dvs_clib.h>
#include <dvs_fifo.h>
}

namespace RavlN {
  template<class DataT> class DListC; }

namespace RavlImageN {


  //! userlevel=Develop
  //: Clip station control class.
  
  class ClipStationProDeviceC {
  public:
    ClipStationProDeviceC(const StringC &devName,const ImageRectangleC &nrect);
    //: Constructor.
    //:!param devName - The name of the device, typical form is "PCI,card:0", "PCI,card:1"
    //:!param nrect   - This parameter is unused

    ~ClipStationProDeviceC();
    //: Destructor.
    
    bool Init();
    //: Setup video modes to a suitable default.
    
    bool GetFrame(void *buff,int x,int y);
    //: Get one frame of video.
    
    BufferC<ByteYUV422ValueC> GetFrame();
    //: Get one field of video.
    
    bool PutFrame(void *buff,int x,int y);
    //: Put one frame of video to the output of the card .
    
    bool CSPGetAttr(const StringC &attrName,StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    //!param: attrName="timecode" - returns the timecode for the last grabbed frame 
    //!param: attrName="FrameBufferSize" - returns the current size of the frame buffer 

    
    bool CSPSetAttr(const StringC &attrName,const StringC &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    //!param: attrName="FrameBufferSize" - returns the current size of the frame buffer 

    bool CSPGetAttrList(DListC<StringC>  & attrList ) const ; 
    //: Get a list of available attributes
    // The list will inherit attributes from parent classes too


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
