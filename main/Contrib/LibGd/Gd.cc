// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id: JasperFormat.cc 5079 2005-08-12 11:32:16Z hickson $"
//! lib=RavlImgGd

#include "Ravl/Image/Gd.hh"

#define DEBUG 0
#if DEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN
{

  GdImageC::GdImageC(const IntT x, const IntT y)
  {
    m_gdImagePtr = gdImageCreateTrueColor(x, y);
  }
  
  void GdImageC::Copy(ImageC<ByteRGBValueC> &image)
  {
    IntT xmax = (static_cast<IntT>(image.Cols()) < m_gdImagePtr->sx ? static_cast<IntT>(image.Cols()) : m_gdImagePtr->sx);
    IntT ymax = (static_cast<IntT>(image.Rows()) < m_gdImagePtr->sy ? static_cast<IntT>(image.Rows()) : m_gdImagePtr->sy);
    
    // Transform from RGB to BGRA
    for (IntT row = 0; row < ymax; row++)
    {
      ByteRGBValueC *srcPtr = image.Row(row);
      int *dstPtr = m_gdImagePtr->tpixels[row];
      for (IntT col = 0; col < xmax; col++)
      {
        int colIndex = gdImageColorAllocate(m_gdImagePtr, srcPtr->Red(), srcPtr->Green(), srcPtr->Blue());
        *dstPtr = colIndex;
        srcPtr++;
        dstPtr++;
      }
    }
  }
  
  ImageC<ByteRGBValueC> GdImageC::GetImage()
  {
    ImageC<ByteRGBValueC> image(m_gdImagePtr->sy, m_gdImagePtr->sx);

    // Transform from BGRA to RGB
    for (IntT row = 0; row < m_gdImagePtr->sy; row++)
    {
      char *srcPtr = reinterpret_cast<char*>(m_gdImagePtr->tpixels[row]);
      ByteRGBValueC *dstPtr = image.Row(row);
      for (IntT col = 0; col < m_gdImagePtr->sx; col++)
      {
        *dstPtr = ByteRGBValueC(srcPtr[2], srcPtr[1], srcPtr[0]);
        srcPtr += 4;
        dstPtr++;
      }
    }
    
    return image;
  }
  
  GdImageC::~GdImageC()
  {
    gdImageDestroy(m_gdImagePtr);
  }
  
}
