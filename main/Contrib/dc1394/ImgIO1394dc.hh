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
#include "Ravl/DP/AttributeType.hh"
#include "Ravl/Hash.hh"
#include "Ravl/Tuple2.hh"
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

    bool HandleGetAttr(const StringC &attrName,StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleSetAttr(const StringC &attrName,const StringC &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleGetAttr(const StringC &attrName,IntT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleSetAttr(const StringC &attrName,const IntT &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleGetAttr(const StringC &attrName,RealT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleSetAttr(const StringC &attrName,const RealT &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleGetAttr(const StringC &attrName,bool &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleSetAttr(const StringC &attrName,const bool &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    RealT SetFrameRate(RealT speed);
    //: Set capture framerate.
    // Returns the actual frame rate.
    
  protected:
    void BuildAttrList(AttributeCtrlBodyC &attrCtrl);
    //: Build attribute list.
    
    enum ControlTypeT { CT_IntValue,CT_FloatValue,CT_OnOff,CT_Auto };
    
    HashC<StringC,Tuple2C<IntT,ControlTypeT> > name2featureid;
    
    raw1394handle_t raw1394handle;
    dc1394_cameracapture camera;
    nodeid_t cameraNode;
    
    StringC camera_vendor;
    StringC camera_model;
    StringC camera_euid;
    
    int cam_channel;
    int cam_format;
    int cam_mode;
    int cam_speed;
    int cam_framerate;
    quadlet_t available_framerates;
  };
  
  template<typename PixelT>
  class DPIImage1394dcBodyC
    : public DPIPortBodyC<ImageC<PixelT> >,
      public ImgIO1394dcBaseC
  {
  public:
    DPIImage1394dcBodyC(const StringC &dev)
      : ImgIO1394dcBaseC()
    { 
      Open(dev,typeid(PixelT)); 
      BuildAttrList(*this);
    }
    //: Constructor.
    
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

    virtual bool GetAttr(const StringC &attrName,StringC &attrValue){ 
      if(HandleGetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const StringC &attrValue) { 
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    
    virtual bool GetAttr(const StringC &attrName,IntT &attrValue){ 
      if(HandleGetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const IntT &attrValue) { 
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool GetAttr(const StringC &attrName,RealT &attrValue){ 
      if(HandleGetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const RealT &attrValue) { 
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool GetAttr(const StringC &attrName,bool &attrValue){ 
      if(HandleGetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const bool &attrValue) { 
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
  protected:
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
