// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlIOV4L2
//! author = "Warren Moore"

#include "Ravl/Image/IOV4L2.hh"
#include "Ravl/DP/AttributeValueTypes.hh"
#include "Ravl/Image/V4L2Buffer.hh"
#include "Ravl/Array2dIter.hh"

#include <linux/videodev.h>
#include <linux/types.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define USE_V4L2 1

#define CHAR_STREAM_FROM_4CC(n) \
     (char)((n      ) & 0xff) \
  << (char)((n >>  8) & 0xff) \
  << (char)((n >> 16) & 0xff) \
  << (char)((n >> 24) & 0xff)

namespace RavlImageN
{
  
  typedef struct SupportedFormatT
  {
    const type_info &m_objectType;          //: Ref to type id
    UIntT m_pixelFormat;                    //: 4CC of required capture mode

    SupportedFormatT(const type_info &objectType, const UIntT pixelFormat) :
      m_objectType(objectType),
      m_pixelFormat(pixelFormat)
    {
    }
    //: Mmmm, structure constructor
  };
  //: Structure used to store supported pixel formats



  const static UIntT g_supportedFormats = 2;
  const static SupportedFormatT g_supportedFormat[g_supportedFormats] =
  {
    SupportedFormatT(typeid(ImageC<ByteRGBValueC>), v4l2_fourcc('B', 'G', 'R', '4')),
    SupportedFormatT(typeid(ImageC<ByteT        >), v4l2_fourcc('G', 'R', 'E', 'Y')),
  };
  //: Mapping from image type id required V4L2 capture mode.
  // Note: Only supports a 1:1 mapping at the moment, this should be changed to support more hardware



  const static UIntT g_defaultWidth = 320;
  //: Default capture width (usually overridden by getting with initial format)

  const static UIntT g_defaultHeight = 240;
  //: Default capture height (usually overridden by getting with initial format)

  const static UIntT g_defaultBuffers = 3;
  //: Default number of capture buffers
  
  const static UIntT g_defaultField = V4L2_FIELD_ANY; 
  //: Default captured field ordering
  
  
  
  IOV4L2BaseC::IOV4L2BaseC(const StringC &device, const UIntT channel, const type_info &pixelType) :
    m_pixelType(pixelType),
    m_device(device),
    m_channel(channel),
    m_fd(-1),
    m_width(g_defaultWidth),
    m_height(g_defaultHeight),
    m_fieldFormat(g_defaultField),
    m_bufferMax(g_defaultBuffers),
    m_bufferCount(0),
    m_bufferOut(0),
    m_buffers(NULL),
    m_seqNum(-1)
  {
    // Open the device
    Open(device, channel);
    
    // Check the format
    if (IsOpen())
    {
      // Check for a supported format
      if (CheckFormat(pixelType))
      {
        // Do final initialisation...
      }
      else
      {
        // Failed to find supported format
        ONDEBUG(cerr << "IOV4L2BaseC::IOV4L2BaseC unsupported image format" << endl;)
        Close();
      }
    }
  }
  
  
  
  IOV4L2BaseC::~IOV4L2BaseC()
  {
    // Release, if configured
    if (IsConfigured())
      ReleaseCapture();
    
    // Close, if open
    if (IsOpen())
      Close();
  }
  
  
  
  bool IOV4L2BaseC::GetFrame(ImageC<ByteRGBValueC> &img, IOV4L2C<ByteRGBValueC> parent)
  {
    RavlAssertMsg(IsOpen() && IsConfigured(), "IOV4L2BaseC::GetFrame<ByteRGBValueC> device not ready");
    RavlAssertMsg(m_bufferOut <= m_bufferCount, "IOV4L2BaseC::GetFrame<ByteRGBValueC> dequeued buffers exceed allocated count");
    
    // Dequeue a buffer
    v4l2_buffer buffer;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_DQBUF, &buffer) == -1)
    {
      cerr << "IOV4L2BaseC::GetFrame<ByteRGBValueC> ioctl(VIDIOC_DQBUF) failed" << endl;
      return false;
    }
    m_bufferOut++;

    ONDEBUG( \
      cerr << "IOV4L2BaseC::GetFrame<ByteRGBValueC> ioctl(VIDIOC_DQBUF)" << endl; \
      cerr << "  index(" << buffer.index << ")" << endl; \
      cerr << "  field(" << buffer.field << ")" << endl; \
      cerr << "  seq(" << buffer.sequence << ")" << endl; \
    )
    
    // Create the image
    img = ImageC<ByteRGBValueC>(m_height, m_width);
    ByteT *iData = (ByteT*)m_buffers[buffer.index].m_start;
    for(Array2dIterC<ByteRGBValueC> it(img); it; it++)
    {
      it.Data() = ByteRGBValueC(iData[0], iData[1], iData[2]);
      iData += 4;
    }

    // Manually release the buffer
    parent.ReleaseBuffer(buffer.index);
    
    return true;
  }



  bool IOV4L2BaseC::GetFrame(ImageC<ByteT> &img, IOV4L2C<ByteT> parent)
  {
    RavlAssertMsg(IsOpen() && IsConfigured(), "IOV4L2BaseC::GetFrame<ByteT> device not ready");
    RavlAssertMsg(m_bufferOut <= m_bufferCount, "IOV4L2BaseC::GetFrame<ByteT> dequeued buffers exceed allocated count");
    
    // Dequeue a buffer
    v4l2_buffer buffer;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_DQBUF, &buffer) == -1)
    {
      cerr << "IOV4L2BaseC::GetFrame<ByteT> ioctl(VIDIOC_DQBUF) failed" << endl;
      return false;
    }
    m_bufferOut++;

    ONDEBUG( \
      cerr << "IOV4L2BaseC::GetFrame<ByteT> ioctl(VIDIOC_DQBUF)" << endl; \
      cerr << "  index(" << buffer.index << ")" << endl; \
      cerr << "  field(" << buffer.field << ")" << endl; \
      cerr << "  seq(" << buffer.sequence << ")" << endl; \
    )
    
    // Create the fast buffer image
    {
      RavlAssertMsg(buffer.memory == V4L2_MEMORY_MMAP, "IOV4L2BaseC::GetFrame<ByteT> buffer not mmap-ed");
      ImageC<ByteT> wrappedImg(m_height, m_width, V4L2BufferC<ByteT>(parent, buffer.index, (ByteT*)m_buffers[buffer.index].m_start, (UIntT)m_buffers[buffer.index].m_length));
      
      // Select the final image
      img = ImageC<ByteT>(wrappedImg.Copy());
    }

    return true;
  }



  void IOV4L2BaseC::ReleaseBuffer(const UIntT index)
  {
    RavlAssertMsg(IsOpen() && IsConfigured(), "IOV4L2BaseC::ReleaseFrame device not ready");
    RavlAssertMsg(m_bufferOut > 0, "IOV4L2BaseC::ReleaseFrame no dequeued buffers");
    
    // Query the buffer
    v4l2_buffer buffer;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.index = index;
    if (ioctl(m_fd, VIDIOC_QUERYBUF, &buffer) == -1)
    {
      cerr << "IOV4L2BaseC::ReleaseBuffer ioctl(VIDIOC_QUERYBUF) failed(" << index << ")" << endl;
      return;
    }
  
    // Re-queue the buffer
    if (ioctl(m_fd, VIDIOC_QBUF, &buffer) == -1)
    {
      cerr << "IOV4L2BaseC::ReleaseBuffer ioctl(VIDIOC_QBUF) failed(" << index << ")" << endl;
    }
    else
    {
      ONDEBUG(cerr << "IOV4L2BaseC::ReleaseBuffer ioctl(VIDIOC_QBUF) requeued(" << index << ")" << endl;)
      m_bufferOut--;
    }
  }
  
  bool IOV4L2BaseC::Open(const StringC &device, const UIntT channel)
  {
    RavlAssertMsg(!IsOpen(), "IOV4L2BaseC::Open called on open device");
    
    // Open the device
    m_fd = open(device, O_RDWR);
    
    // Reset the params
    m_width = g_defaultWidth;
    m_height = g_defaultHeight;
    m_seqNum = -1;
    
    return (m_fd != -1);
  }
  
  
  
  void IOV4L2BaseC::Close()
  {
    RavlAssertMsg(!IsConfigured(), "IOV4L2BaseC::Close called on closed device");
    RavlAssertMsg(IsOpen(), "IOV4L2BaseC::Close called on closed device");
    
    // All done, close the device
    close(m_fd);
  }
  
  
  
  bool IOV4L2BaseC::CheckFormat(const type_info &pixelType)
  {
    RavlAssertMsg(IsOpen(), "IOV4L2BaseC::Open called without open device");
    
    // Is capture supported?
    v4l2_capability cap;
    if (ioctl(m_fd, VIDIOC_QUERYCAP, &cap) != -1)
    {
      ONDEBUG( \
        cerr << "IOV4L2BaseC::CheckFormat device(" << m_device << ")" << endl; \
        cerr << "  ioctl(VIDIOC_QUERYCAP) driver (" << cap.driver << ")" << endl; \
        cerr << "  ioctl(VIDIOC_QUERYCAP) card (" << cap.card << ")" << endl; \
        cerr << "  ioctl(VIDIOC_QUERYCAP) bus (" << cap.bus_info << ")" << endl; \
      )
    }
    else
    {
      ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat ioctl(VIDIOC_QUERYCAP) device(" << m_device << ") capture not supported" << endl;)
      return false;
    }
    
    // Enumerate the inputs
    UIntT inputCount = 0;
    v4l2_input input;
    input.index = 0;
    ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat ioctl(VIDIOC_ENUMINPUT)" << endl;)
    while (ioctl(m_fd, VIDIOC_ENUMINPUT, &input) != -1)
    {
      ONDEBUG(cerr << "  [" << inputCount << "] name(" << input.name << ")" << endl;)
      inputCount++;

      if (input.index == m_channel)
      {
        ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat channel supported by device" << endl;)
        break;
      }

      input.index = inputCount;
    }

    // Check for a valid input
    if (m_channel >= inputCount)
    {
      cerr << "IOV4L2BaseC::CheckFormat channel(" << m_channel << ") not supported" << endl;
      return false;
    }
    
    // Get the current settings format
    v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_pix_format *pfmt = (v4l2_pix_format*)&(fmt.fmt);
    if (ioctl(m_fd, VIDIOC_G_FMT, &fmt) != -1)
    {
      ONDEBUG( \
        cerr << "IOV4L2BaseC::CheckFormat ioctl(VIDIOC_G_FMT)" << endl; \
        cerr << "  width(" << pfmt->width << ")" << endl; \
        cerr << "  height(" << pfmt->height << ")" << endl; \
        cerr << "  4cc(" << CHAR_STREAM_FROM_4CC(pfmt->pixelformat) << ")" << endl; \
        cerr << "  field(" << pfmt->field << ")" << endl; \
      )
      
      // Store the default width and height
      m_width = pfmt->width;
      m_height = pfmt->height;
    }
    
    // Which format do I need to support this pixel format?
    UIntT pixelIndex = 0;
    ONDEBUG( \
      cerr << "IOV4L2BaseC::CheckFormat typeid(" << pixelType.name() << ")" << endl; \
    )
    while (pixelIndex < g_supportedFormats)
    {
      // Search the table
      if (pixelType == g_supportedFormat[pixelIndex].m_objectType)
      {
        ONDEBUG( \
          cerr << "IOV4L2BaseC::CheckFormat requires format(" << CHAR_STREAM_FROM_4CC(g_supportedFormat[pixelIndex].m_pixelFormat) << ")" << endl; \
        )
        m_pixelFormat = g_supportedFormat[pixelIndex].m_pixelFormat;
        break;
      }
      
      // Try the next entry
      pixelIndex++;
    }
    if (pixelIndex == g_supportedFormats)
    {
      ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat no suitable pixel format supported" << endl;)
      return false;
    }
    
    // Enumerate the capture formats
    bool supported = false;
    v4l2_fmtdesc desc;
    desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    desc.index = 0;
    ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat ioctl(VIDIOC_ENUM_FMT)" << endl;)
    while (ioctl(m_fd, VIDIOC_ENUM_FMT, &desc) != -1)
    {
      ONDEBUG( \
        cerr << "  [" << desc.index << "] ";
        cerr << "4cc(" << CHAR_STREAM_FROM_4CC(desc.pixelformat) << ") "; \
        cerr << "desc(" << desc.description << ")" << endl; \
      )
      desc.index++;
      
      // Check the pixel format is supported
      if (desc.pixelformat == m_pixelFormat)
      {
        ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat pixel format supported by device" << endl;)
        supported = true;
        break;
      }
    }
    
    return supported;
  }
  
  
  
  bool IOV4L2BaseC::ConfigureCapture()
  {
    RavlAssertMsg(IsOpen(), "IOV4L2BaseC::ConfigureCapture device not open");
    RavlAssertMsg(!IsConfigured(), "IOV4L2BaseC::ConfigureCapture device already configured");

    // Set the input channel
    if (ioctl(m_fd, VIDIOC_S_INPUT, &m_channel) == -1)
    {
      cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_S_INPUT) invalid input channel(" << m_channel << ")" << endl;
      return false;
    }

    // Get the current format
    v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_pix_format *pfmt = (v4l2_pix_format*)&(fmt.fmt);
    if (ioctl(m_fd, VIDIOC_G_FMT, &fmt) == -1)
    {
      ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture unable to obtain current capture format" << endl;)
      return false;
    }
    
    // Set the capture mode 
    pfmt->width = m_width;
    pfmt->height = m_height;
    pfmt->pixelformat = m_pixelFormat;
    pfmt->field = (v4l2_field)m_fieldFormat;
    if (ioctl(m_fd, VIDIOC_S_FMT, &fmt) == -1)
    {
      ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture unable to set capture format" << endl;)
      return false;
    }
    
    // Request the required amount of memory-mapped buffers
    v4l2_requestbuffers reqbuf;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    reqbuf.count = m_bufferMax;
    if (ioctl(m_fd, VIDIOC_REQBUFS, &reqbuf) != -1)
    {
      if (m_bufferMax != reqbuf.count)
      {
        cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_REQBUFS) requested(" << m_bufferMax << ") buffers, got(" << reqbuf.count << ")" << endl;
      }
      else
      {
        ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_REQBUFS) requested(" << m_bufferMax << ") buffers, got(" << reqbuf.count << ")" << endl;)
      }
      
      if (reqbuf.count <= 0)
      {
        ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_REQBUFS) unable to allocate any mmap-ed buffers" << endl;)
        return false;
      }

      // Allocate the buffer table
      m_buffers = (TBuf*)calloc(reqbuf.count, sizeof(*m_buffers));
      RavlAssertMsg(m_buffers != NULL, "IOV4L2BaseC::ConfigureCapture failed to allocate buffers");

      // Configure each buffers mmap entry
      for (m_bufferCount = 0; m_bufferCount < reqbuf.count; m_bufferCount++)
      {
        // Query the buffer
        v4l2_buffer buffer;
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.index = m_bufferCount;
        if (ioctl(m_fd, VIDIOC_QUERYBUF, &buffer) == -1)
        {
          ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_QUERYBUF) failed buffer(" << m_bufferCount << ")" << endl;)
          break;
        }
      
        // Map the buffer
        m_buffers[m_bufferCount].m_length = buffer.length;
        m_buffers[m_bufferCount].m_start = mmap(NULL,
                                                buffer.length,
                                                PROT_READ | PROT_WRITE, // required
                                                MAP_SHARED,             // recommended
                                                m_fd,
                                                buffer.m.offset);
      
        // Verify
        if (m_buffers[m_bufferCount].m_start == MAP_FAILED)
        {
          ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture mmap failed buffer(" << m_bufferCount << ")" << endl;)
          break;
        }
        
        // Queue the buffer
        if (ioctl(m_fd, VIDIOC_QBUF, &buffer) == -1)
        {
          ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_QBUF) failed buffer(" << m_bufferCount << ")" << endl;)
          break;
        }
      }

      if (reqbuf.count != m_bufferCount)
      {
        cerr << "IOV4L2BaseC::ConfigureCapture allocated(" << reqbuf.count << ") buffers, mmap-ed(" << m_bufferCount << ") buffers" << endl;
      }
      else
      {
        ONDEBUG(cerr << "IOV4L2BaseC::ConfigureCapture allocated(" << reqbuf.count << ") buffers, mmap-ed(" << m_bufferCount << ") buffers" << endl;)
      }
    }

    // Stream-on, Daniel-san
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_STREAMON, &type) == -1)
    {
      cerr << "IOV4L2BaseC::ConfigureCapture ioctl(VIDIOC_STREAMON) failed (" << errno << ")" << endl;
      ReleaseCapture();
      return false;
    }
    
    return true;
  }
  
  
  
  void IOV4L2BaseC::ReleaseCapture()
  {
    RavlAssertMsg(m_bufferOut == 0, "IOV4L2BaseC::ReleaseCapture called with fast-buffers still actives");
    RavlAssertMsg(IsConfigured(), "IOV4L2BaseC::ReleaseCapture called on unconfigured device");
    RavlAssertMsg(IsOpen(), "IOV4L2BaseC::ReleaseCapture called on closed device");
    
    // Stream-off, Daniel-san
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_STREAMOFF, &type) == -1)
    {
      cerr << "IOV4L2BaseC::ReleaseCapture ioctl(VIDIOC_STREAMOFF) failed" << endl;
    }

    // Unmap the buffers
    RavlAssertMsg(m_buffers != NULL, "IOV4L2BaseC::ReleaseCapture null buffer pointer");
    for (UIntT i = 0; i < m_bufferCount; i++)
      munmap(m_buffers[i].m_start, m_buffers[i].m_length);
    m_bufferCount = 0;
    free(m_buffers);
    m_buffers = NULL;
  }



  bool IOV4L2BaseC::HandleGetAttr(const StringC &attrName, StringC &attrValue)
  {
    return false;
  }
  
  
  
  bool IOV4L2BaseC::HandleSetAttr(const StringC &attrName, const StringC &attrValue)
  {
    return false;
  }
  
  
  
  bool IOV4L2BaseC::HandleGetAttr(const StringC &attrName, IntT &attrValue)
  {
    return false;
  }
  
  
  
  bool IOV4L2BaseC::HandleSetAttr(const StringC &attrName, const IntT &attrValue)
  {
    return false;
  }
  
  

  bool IOV4L2BaseC::BuildAttributes(AttributeCtrlBodyC &attrCtrl)
  {
//    attrCtrl.RegisterAttribute(AttributeTypeNumC<IntT>("width",         "Width",       true, true,    -1, 65000, 1, -1));

    return true;
  }
}

