// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_IOV4L2_HEADER
#define RAVL_IOV4L2_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlIOV4L2
//! author = "Warren Moore"

#include "Ravl/DP/Port.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"

namespace RavlImageN
{
  using namespace RavlN;
  
  class IOV4L2BaseC
  {
  public:
    IOV4L2BaseC(const StringC &device, const UIntT channel, const type_info &pixelType);
    //: Constructor
    
    ~IOV4L2BaseC();
    //: Destructor
    
    bool GetFrame(ImageC<ByteRGBValueC> &img);
    //: Get next image.

    bool GetFrame(ImageC<ByteT> &img);
    //: Get next image.

    UIntT LastFrameNum() const
    { return m_seqNum; }
    //: Get the last frame sequence number
    
    StreamPosT LastFrameSeq() const
    { return m_seqNum; }
    //: Get the last frame sequence number
    
    bool IsOpen() const
    { return (m_fd != -1); }
    //: Is the device open?
    
  protected:
    bool Open(const StringC &device, const UIntT channel);
    //: Open the device
    
    void Close();
    //: Close the device
    
    bool CheckFormat(const type_info &pixelType);
    //: Check if the pixel type is supported
    
  protected:
    StringC m_device;                   // Device name
    UIntT m_channel;                    // Channel number
    int m_fd;                           // File descriptor
    StreamPosT m_seqNum;                // Last sequence number
  };
  
  template <class PixelT>
  class IOV4L2BodyC :
    public DPIPortBodyC< ImageC<PixelT> >,
    public IOV4L2BaseC
  {
  public:
    IOV4L2BodyC(const StringC &device, const UIntT channel) :
      IOV4L2BaseC(device, channel, typeid(PixelT))
    {
    }
    //: Constructor.
    
    virtual ~IOV4L2BodyC()
    {
    }
    //: Destructor.
    
    virtual ImageC<PixelT> Get()
    {
      ImageC<PixelT> img;
      if(!Get(img))
        throw DataNotReadyC("Failed to get next frame.");
      return img;
    }
    //: Get next frame.
    
    virtual bool Get(ImageC<PixelT> &img)
    { return GetFrame(img); }
    //: Get next image.
    
    virtual UIntT Tell() const
    { return LastFrameSeq(); }
    //: Find current location in stream.
    
    virtual StreamPosT Tell64() const
    { return LastFrameSeq(); }
    //: Find current location in stream.
    
    virtual bool IsGetEOS() const
    {
      return false;
    }
    //: Is it the EOS

    virtual bool GetAttr(const StringC &attrName, StringC &attrValue)
    {
      return false;
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.

  protected:
    void BuildAttributes()
    {
    }
    //: Register stream attributes
  };

  template <class PixelT>
  class IOV4L2C :
    public DPIPortC< ImageC<PixelT> >
  {
  public:
    IOV4L2C() :
      DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    IOV4L2C(const StringC &device, const UIntT channel) :
      DPEntityC(*new IOV4L2BodyC<PixelT>(device, channel))
    {}
    //: Constructor.

  protected:
    IOV4L2BodyC<PixelT> &Body()
    { return static_cast<IOV4L2BodyC<PixelT> &>(DPIPortC< ImageC<PixelT> >::Body()); }
    //: Access body.

    const IOV4L2BodyC<PixelT> &Body() const
    { return static_cast<const IOV4L2BodyC<PixelT> &>(DPIPortC< ImageC<PixelT> >::Body()); }
    //: Access body.
  };
}

#endif
