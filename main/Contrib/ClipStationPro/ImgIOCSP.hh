// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_IMGIOCSP_HEADER
#define RAVLIMAGE_IMGIOCSP_HEADER 1
//! rcsid="$Id$"
//! lib=CSPDriver
//! docentry="Ravl.Contrib.Image IO.ClipStationPro"
//! author="Charles Galambos"
//! file="Contrib/ClipStationPro/ImgIOCSP.hh"

#include "Ravl/Image/CSPControl.hh"
#include "Ravl/DP/Port.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Image/YUV422Value.hh"
#include "Ravl/Image/Deinterlace.hh"

namespace RavlImageN {
  
  //! userlevel=Develop
  //: ClipStationPro frame grabber.
  
  template<class PixelT>
  class DPIImageClipStationProBodyC
    : public DPIPortBodyC<ImageC<PixelT> >,
      public ClipStationProDeviceC
  {
  public:
    DPIImageClipStationProBodyC(const StringC &dev,const ImageRectangleC &nrect = ImageRectangleC(0,-1,0,-1))
      : ClipStationProDeviceC(dev,nrect) //,typeid(PixelT),nrect)
    {}
    //: Constructor.
    
    virtual ImageC<PixelT> Get() {
      BufferC<PixelT> buf = GetFrame();
      if(!buf.IsValid())
	return ImageC<PixelT>();
      ImageC<PixelT> img((UIntT) rect.Rows(),(UIntT) rect.Cols(),buf);
      return Deinterlace(img);
    }
    //: Get next image.
    
    virtual bool Get(ImageC<PixelT> &buff) {
      BufferC<PixelT> buf = GetFrame();
      if(!buf.IsValid())
	return false;
      ImageC<PixelT> img((UIntT) rect.Rows(),(UIntT) rect.Cols(),buf);
      buff = Deinterlace(img);
      return true;
    }
    //: Get next image.
    
    virtual bool IsGetReady() const
    { return true; }
    //: Is some data ready ?
    // true = yes.
    // Defaults to !IsGetEOS().
    
    virtual bool IsGetEOS() const
    { return dev == 0; }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    virtual bool GetAttr(const StringC &attrName,StringC &attrValue) {
      return CSPGetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const StringC &attrValue) {
      return CSPSetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

  };
  
  //! userlevel=Normal
  //: ClipStationPro frame grabber.
  
  template<class PixelT>
  class DPIImageClipStationProC
    : public DPIPortC<ImageC<PixelT> >
  {
  public:
    DPIImageClipStationProC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    // creates an invalid handle.
    
    DPIImageClipStationProC(const StringC &dev,const ImageRectangleC &nrect = ImageRectangleC(0,-1,0,-1))
      : DPEntityC(*new DPIImageClipStationProBodyC<PixelT>(dev,nrect))
    {}
    //: Constructor.
    
  protected:
    DPIImageClipStationProBodyC<PixelT> &Body()
    { return dynamic_cast<DPIImageClipStationProBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.

    const DPIImageClipStationProBodyC<PixelT> &Body() const
    { return dynamic_cast<const DPIImageClipStationProBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
  public:  
    
  };
  
#if 0
  
  //! userlevel=Normal
  //: ClipStationPro frame grabber.
  
  template<class PixelT>
  class DPIImageClipStationProC
    : public DPIPortC<ImageC<PixelT> >
  {
  public:
    DPIImageClipStationProC()
      : DPEntityC(true)
    {}
  //: Default constructor.
  // creates an invalid handle.
  
    DPIImageClipStationProC(const StringC &dev,const ImageRectangleC &nrect = ImageRectangleC(0,-1,0,-1))
      : DPEntityC(*new DPIImageClipStationProBodyC<PixelT>(dev,nrect))
    {}
    //: Constructor.
    
  protected:
    DPIImageClipStationProBodyC<PixelT> &Body()
    { return dynamic_cast<DPIImageClipStationProBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPIImageClipStationProBodyC<PixelT> &Body() const
    { return dynamic_cast<const DPIImageClipStationProBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
  public:  
    
    const char *ParamName(DPIImageBaseClipStationProBodyC::VideoParamT pr) const
    { return Body().ParamName(pr); }
    //: Get the name of each paramiter.
    
    int GetParam(DPIImageBaseClipStationProBodyC::VideoParamT pr) const
    { return Body().GetParam(pr); }
    //: Get paramiter value
    
    bool SetParam(DPIImageBaseClipStationProBodyC::VideoParamT pr,int val)
    { return Body().SetParam(pr,val); }
    //: Setup paramiter value.
    
    void DumpParam(ostream &out) const
    { return Body().DumpParam(out); }
    //: Dump current settings to 'out'.
    
  };
#endif

}


#endif
