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
  
  IOV4L2BaseC::IOV4L2BaseC(const StringC &device, const UIntT channel, const type_info &pixelType) :
    m_device(device),
    m_channel(channel),
    m_fd(-1),
    m_seqNum(-1)
  {
    // Open the device
    Open(device, channel);
    
    // Check the format
    if (IsOpen())
      if (!CheckFormat(pixelType))
      {
        ONDEBUG(cerr << "IOV4L2BaseC::IOV4L2BaseC unsupported image format" << endl;)
        Close();
      }
  }
  
  IOV4L2BaseC::~IOV4L2BaseC()
  {
    // Close, if open
    if (IsOpen())
      Close();
  }
  
  bool IOV4L2BaseC::GetFrame(ImageC<ByteRGBValueC> &img)
  {
    return false;
  }

  bool IOV4L2BaseC::GetFrame(ImageC<ByteT> &img)
  {
    return false;
  }
    
  bool IOV4L2BaseC::Open(const StringC &device, const UIntT channel)
  {
    RavlAssertMsg(m_fd == -1, "IOV4L2BaseC::Open called on open device");
    
    // Open the device
    m_fd = open(device, O_RDWR);
    
    // Reset the params
    m_seqNum = -1;
    
    return (m_fd != -1);
  }
    
  void IOV4L2BaseC::Close()
  {
    if (m_fd != -1)
    {
      // All done, close the device
      close(m_fd);
    }
  }
  
  bool IOV4L2BaseC::CheckFormat(const type_info &pixelType)
  {
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
    if (ioctl(m_fd, VIDIOC_G_FMT, &fmt) != -1)
    {
      ONDEBUG( \
        v4l2_pix_format *pfmt = (v4l2_pix_format*)&(fmt.fmt); \
        cerr << "IOV4L2BaseC::CheckFormat ioctl(VIDIOC_G_FMT)" << endl; \
        cerr << "  width(" << pfmt->width << ")" << endl; \
        cerr << "  height(" << pfmt->height << ")" << endl; \
        cerr << "  4cc(" << CHAR_STREAM_FROM_4CC(pfmt->pixelformat) << ")" << endl; \
        cerr << "  field(" << pfmt->field << ")" << endl; \
      )
    }
    
    // Which format do I need to support this pixel format?
    UIntT pixelIndex = 0;
    UIntT pixelFormat = 0;
    while (pixelIndex < g_supportedFormats)
    {
      // Search the table
      if (pixelType == g_supportedFormat[pixelIndex].m_objectType)
      {
        ONDEBUG( \
          cerr << "IOV4L2BaseC::CheckFormat typeid(" << g_supportedFormat[pixelIndex].m_objectType.name() << ")" << endl; \
          cerr << "  requires format(" << CHAR_STREAM_FROM_4CC(g_supportedFormat[pixelIndex].m_pixelFormat); \
          cerr << ")" << endl; \
        )
        pixelFormat = g_supportedFormat[pixelIndex].m_pixelFormat;
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
      if (desc.pixelformat == pixelFormat)
      {
        ONDEBUG(cerr << "IOV4L2BaseC::CheckFormat pixel format supported by device" << endl;)
        supported = true;
        break;
      }
    }
    
    return supported;
  }
}

