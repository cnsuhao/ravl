// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_IMGIO1394DC_HEADER
#define RAVLIMAGE_IMGIO1394DC_HEADER 1
//! rcsid="$Id$"
//! lib=RavlImgIO1394dc
//! author="Charles Galambos"
//! docentry="Ravl.Contrib.Video_IO.IIDC"

#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/Image.hh"

#include <libdc1394/dc1394_control.h>

namespace RavlImageN {
  
  //! userlevel=Develop
  //: Firewire dc
  
  class ImgIO1394dcBaseC {
  public:
    ImgIO1394dcBaseC();
    //: Constructor.
    
    ~ImgIO1394dcBaseC();
    //: Destructor.
    
    bool Open(const StringC &dev,const type_info &npixType);
    //: Open camera on device.
    
    bool CaptureImage(ImageC<ByteT> &img);
    //: Capture an image.
    
  protected:
    raw1394handle_t raw1394handle;
    dc1394_cameracapture camera;
  };
  
  template<typename PixelT>
  class DPIImage1394dcBodyC
    : public DPIPortBodyC<ImageC<PixelT> >,
      public ImgIO1394dcBaseC
  {
  public:
    DPIImage1394dcBodyC(const StringC &dev)
      : ImgIO1394dcBaseC()
    { Open(dev,typeid(PixelT)); }
    
    virtual bool IsGetReady() const
    { return raw1394handle != 0; }
    //: Is some data ready ?
    // true = yes.
    // Defaults to !IsGetEOS().
    
    virtual bool IsGetEOS() const
    { return !IsGetReady(); }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    virtual bool Get(ImageC<PixelT> &buff) 
    { return CaptureImage(buff); }
    //: Get next image.

    virtual ImageC<PixelT> Get() {
      ImageC<PixelT> buff;
      if(!CaptureImage(buff))
	throw DataNotReadyC("Failed to capture image. ");
      return buff;
    }
    //: Get next image.
    
  };

  template<class PixelT>
  class DPIImage1394dcC
    : public DPIPortC<ImageC<PixelT> >
  {
  public:
    DPIImage1394dcC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    
    DPIImage1394dcC(const StringC &str)
      : DPEntityC(*new DPIImage1394dcBodyC<PixelT>(str))
    {}
  };
}

#endif
