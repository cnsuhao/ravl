// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_VIDIOV4L_HEADER
#define RAVL_VIDIOV4L_HEADER 1
///////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! docentry="Ravl.Contrib.Image IO.Video4Linux"
//! author="Charles Galambos"
//! lib=RavlImgIOV4L
//! file="Contrib/V4L/ImgIOV4L.hh"

#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteYUVValue.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Stream.hh"

namespace RavlImageN {

  //! userlevel=Develop
  //: Base class for Meteor1 frame grabbers.
  
  class DPIImageBaseV4LBodyC
  {
  public:
    DPIImageBaseV4LBodyC(const StringC &dev,const type_info &npixType,const ImageRectangleC &nrect,int channel = -1);
    //: Constructor.
    
    DPIImageBaseV4LBodyC(const StringC &dev,const type_info &npixType,bool half,int channel = -1);
    //: Constructor.
    
    ~DPIImageBaseV4LBodyC();
    //: Destructor.
    
  protected:
    bool Open(const StringC &dev,const type_info &npixType,const ImageRectangleC &nrect);
    //: Open a video for linux device.
    
    bool Open1(int nfd,const type_info &npixType,const ImageRectangleC &nrect);
    //: Open a video for linux 1 device.
    
    bool Close();
    //: Close connection to meteor.
    
    bool NextPre();
    //: Pre next frame capture.
    
    bool NextPost();
    //: Post next frame capture.
    
    bool NextFrame(ImageC<ByteYUVValueC> &ret);
    //: Get next YUV frame from grabber.
    
    bool NextFrame(ImageC<ByteRGBValueC> &ret);
    //: Get next RGB frame from grabber.

    bool NextFrame(ImageC<ByteT> &ret);
    //: Get next grey level frame from grabber.
    
    bool SetupPhilips();
    //: Setup a philips webcam.
    
    enum SourceTypeT { SOURCE_UNKNOWN , SOURCE_USBWEBCAM_PHILIPS };
    
    SourceTypeT sourceType;
    ImageRectangleC maxRect;// Maximum size of grabber rectangle.
    ImageRectangleC rect; // Requested size of grabber rectangle.
    
    int palette; // Current palette mode.    
    int fd;
    ByteT *buffer;
    ByteT *buf_grey;
    ByteT *buf_u;
    ByteT *buf_v;
    int bufLen;
    bool half; // Attempt to get half size images?
    bool memmap; // Use memory mapping.
    SArray1dC<int> frameOffsets;
    int bufNo;
    int channel;
  };
  
  //! userlevel=Develop
  //: V4L frame grabber.
  
  template<class PixelT>
  class DPIImageV4LBodyC
    : public DPIPortBodyC<ImageC<PixelT> >,
      public DPIImageBaseV4LBodyC
  {
  public:
    DPIImageV4LBodyC(const StringC &dev,const ImageRectangleC &nrect = ImageRectangleC(0,-1,0,-1),int channel = -1)
      : DPIImageBaseV4LBodyC(dev,typeid(PixelT),nrect,channel)
    {}
    //: Constructor.
    
    DPIImageV4LBodyC(const StringC &dev,bool half,int channel = -1)
      : DPIImageBaseV4LBodyC(dev,typeid(PixelT),half,channel)
    {}
    //: Constructor.
    
    virtual ImageC<PixelT> Get() {
      ImageC<PixelT> ret;
      NextFrame(ret);
      return ret;
    }
    //: Get next image.
    
    virtual bool Get(ImageC<PixelT> &buff) 
    { return NextFrame(buff); }
    //: Get next image.
    
    virtual bool IsGetReady() const
    { return fd >= 0; }
    //: Is some data ready ?
    // true = yes.
    // Defaults to !IsGetEOS().
    
    virtual bool IsGetEOS() const
    { return fd < 0; }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
  };
  
  //! userlevel=Normal
  //: V4L frame grabber.
  
  template<class PixelT>
  class DPIImageV4LC
    : public DPIPortC<ImageC<PixelT> >
  {
  public:
    DPIImageV4LC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    // creates an invalid handle.
    
    DPIImageV4LC(const StringC &dev,const ImageRectangleC &nrect = ImageRectangleC(0,-1,0,-1),int channel = -1)
      : DPEntityC(*new DPIImageV4LBodyC<PixelT>(dev,nrect,channel))
    {}
    //: Constructor.
    
    DPIImageV4LC(const StringC &dev,bool half,int channel = -1)
      : DPEntityC(*new DPIImageV4LBodyC<PixelT>(dev,half,channel))
    {}
    //: Constructor.
    
  protected:
    DPIImageV4LBodyC<PixelT> &Body()
    { return dynamic_cast<DPIImageV4LBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPIImageV4LBodyC<PixelT> &Body() const
    { return dynamic_cast<const DPIImageV4LBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
  public:  
    
  };

}
#endif
