// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2005, Charles Galambos.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlMacOSXVideoCapture

#include "Ravl/MacOSX/BufferCVImage.hh"

namespace RavlImageN {

  BufferCVImageBaseC::BufferCVImageBaseC(CVImageBufferRef &aBuffer)
    : m_imageBuffer(aBuffer)
  {
    CVBufferRetain(m_imageBuffer);
    CVPixelBufferLockBaseAddress(m_imageBuffer,0);
  }
  
  // Destructor
  BufferCVImageBaseC::~BufferCVImageBaseC()
  {
    CVPixelBufferUnlockBaseAddress(m_imageBuffer,0);
    CVBufferRelease(m_imageBuffer);  
  }

  // Access start address of buffer.
  void *BufferCVImageBaseC::StartAddress() {
    return CVPixelBufferGetBaseAddress(m_imageBuffer);
  }

  
}
