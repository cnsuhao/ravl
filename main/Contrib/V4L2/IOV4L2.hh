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
  
  template <class PixelT> class IOV4L2C;
  //: Predeclare class handle
  
  /////////////////////////////
  //! userlevel = Develop
  //: V4L2 un-templated body
  
  class IOV4L2BaseC
  {
  public:
    IOV4L2BaseC(const StringC &device, const UIntT channel, const type_info &pixelType);
    //: Constructor
    
    ~IOV4L2BaseC();
    //: Destructor
    
    bool GetFrame(ImageC<ByteRGBValueC> &img, IOV4L2C<ByteRGBValueC> parent);
    //: Get next image.

    bool GetFrame(ImageC<ByteT> &img, IOV4L2C<ByteT> parent);
    //: Get next image.

    void ReleaseBuffer(const UIntT index);
    //: Release a fast buffer.
    // Should not be called directly
    
    UIntT LastFrameNum() const
    { return m_seqNum; }
    //: Get the last frame sequence number
    
    StreamPosT LastFrameSeq() const
    { return m_seqNum; }
    //: Get the last frame sequence number
    
    bool IsOpen() const
    { return (m_fd != -1); }
    //: Is the device open?
    
    bool IsConfigured() const
    { return (m_bufferCount > 0); }
    //: Is the device configured and ready for capture?
    
    bool HandleGetAttr(const StringC &attrName, StringC &attrValue);
    //: Handle get attribute (string)
    // Returns false if the attribute name is unknown.
    
    bool HandleSetAttr(const StringC &attrName, const StringC &attrValue);
    //: Handle set attribute (string)
    // Returns false if the attribute name is unknown.
    
    bool HandleGetAttr(const StringC &attrName, IntT &attrValue);
    //: Handle get attribute (int)
    // Returns false if the attribute name is unknown.
    
    bool HandleSetAttr(const StringC &attrName, const IntT &attrValue);
    //: Handle set attribute (int)
    // Returns false if the attribute name is unknown.
    
    bool BuildAttributes(AttributeCtrlBodyC &attrCtrl);
    //: Build list of attributes.
    
  protected:
    bool Open(const StringC &device, const UIntT channel);
    //: Open the device.
    // Does not configure the capture buffers
    
    void Close();
    //: Close the device.
    // Does not release the capture buffers
    
    bool CheckFormat(const type_info &pixelType);
    //: Check if the pixel type is supported
    
    bool ConfigureCapture();
    //: Configure the capture device, allocating capture buffers
    
    void ReleaseCapture();
    //: Release the capture device, deallocating capture buffers
    
  protected:
    typedef struct
    {
      void *m_start;
      size_t m_length;
    } TBuf;
    //: Buffer mmap data
    
  protected:
  /* Device identification */
    const type_info &m_pixelType;       // Desired image type
    StringC m_device;                   // Device name
    UIntT m_channel;                    // Channel number
    int m_fd;                           // File descriptor

  /* Capture parameters */
    UIntT m_width, m_height;            // Captured size
    UIntT m_pixelFormat;                // Stored pixel format
    UIntT m_fieldFormat;                // Captured field selection

  /* Buffer parameters */
    UIntT m_bufferMax;                  // Number of capture buffers
    UIntT m_bufferCount;                // Number of capture buffers
    UIntT m_bufferOut;                  // Unqueued buffer count
    TBuf *m_buffers;                    // Array of mmap'd buffers
    
  /* Frame attributes */
    StreamPosT m_seqNum;                // Last sequence number
  };
  
  
  
  /////////////////////////////
  //! userlevel = Develop
  //: V4L2 templated body
  
  template <class PixelT>
  class IOV4L2BodyC :
    public DPIPortBodyC< ImageC<PixelT> >,
    public IOV4L2BaseC
  {
  public:
    IOV4L2BodyC(const StringC &device, const UIntT channel) :
      IOV4L2BaseC(device, channel, typeid(ImageC<PixelT>))
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
      if (!Get(img))
        throw DataNotReadyC("Failed to get next frame.");
      return img;
    }
    //: Get next frame.
    
    virtual bool Get(ImageC<PixelT> &img)
    { 
      // Check the device is open
      if (!IsOpen())
        return false;
      
      // Configure the device, if not
      if (!IsConfigured())
        if (!ConfigureCapture())
        {
          Close();
          return false;
        }
        
      return GetFrame(img, IOV4L2C<PixelT>(*this));
    }
    //: Get next image.
    
    virtual UIntT Tell() const
    { return LastFrameSeq(); }
    //: Find current location in stream.
    
    virtual StreamPosT Tell64() const
    { return LastFrameSeq(); }
    //: Find current location in stream.
    
    virtual bool IsGetReady() const
    {
      return IsOpen();
    }
    
    virtual bool IsGetEOS() const
    {
      return !IsOpen();
    }
    //: Is it the EOS

    virtual bool GetAttr(const StringC &attrName, StringC &attrValue)
    {
      if (HandleGetAttr(attrName, attrValue))
        return true;
      return DPPortBodyC::GetAttr(attrName, attrValue);
    }
    //: Handle get attribute (string)
    // Returns false if the attribute name is unknown.
    
    virtual bool SetAttr(const StringC &attrName, const StringC &attrValue)
    {
      if (HandleSetAttr(attrName, attrValue))
        return true;
      return DPPortBodyC::SetAttr(attrName, attrValue);
    }
    //: Handle set attribute (string)
    // Returns false if the attribute name is unknown.
    
    virtual bool GetAttr(const StringC &attrName, IntT &attrValue)
    {
      if (HandleGetAttr(attrName, attrValue))
        return true;
      return DPPortBodyC::GetAttr(attrName, attrValue);
    }
    //: Handle get attribute (int)
    // Returns false if the attribute name is unknown.
    
    virtual bool SetAttr(const StringC &attrName, const IntT &attrValue)
    {
      if (HandleSetAttr(attrName, attrValue))
        return true;
      return DPPortBodyC::SetAttr(attrName, attrValue);
    }
    //: Handle set attribute (int)
    // Returns false if the attribute name is unknown.
  };

  
  
  /////////////////////////////
  //! userlevel = Advanced
  //: V4L2 format
  // BIG OBJECT
  
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

    explicit IOV4L2C(IOV4L2BodyC<PixelT> &body) :
      DPEntityC(body)
    {}
    //: Body constructor.

    void ReleaseBuffer(const UIntT index)
    { Body().ReleaseBuffer(index); }
    //: Release a fast buffer.
    // Should not be called directly
    
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
