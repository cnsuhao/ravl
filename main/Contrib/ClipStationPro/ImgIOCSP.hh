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
//! docentry="Ravl.Images.Video.Video IO.ClipStationPro"
//! author="Charles Galambos"
//! file="Ravl/Contrib/ClipStationPro/ImgIOCSP.hh"
//! example=exCSPGrab.cc
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
    //!param: dev   - The name of the hardware device eg. "PCI,card:0", "PCI,card:1"
    //!param: nrect - The rectangle to capture, by default a zero size image is grabbed. This must be an even number !. Maximum usable area = 576*720  


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

    virtual bool GetAttrList(DListC<StringC> & list) const
    { return CSPGetAttrList (list) ; } 
    //: Append the list of available attributes 
    // Available attributes for this class are appended to the list


  };
  





  //! userlevel=Normal
  //: ClipStationPro frame grabber.
   // This class provides a convenient user interface for access to the ClipStationPro capture cards. 
  // This interface provides attributes such as "timecode" and "FrameBufferSize" which can be accessed/modified through the 
  // GetAttr SetAttr methods. 
  // See the data processing <a href="../Tree/Ravl.Core.Data_Processing.Attributes.html"> attribute handling mechanism. </a>  For more details. 
  // Also see the <a href="../Tree/Ravl.Images.Video.html"> video section </a> for a list of common attributes. 
  // 

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
    //!param: dev   - The name of the hardware device eg. "PCI,card:0", "PCI,card:1"
    //!param: nrect - The rectangle to capture, by default a zero size image is grabbed. This must be an even number !. Maximum usable area = 576*720  

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
