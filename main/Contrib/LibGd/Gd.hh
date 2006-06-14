// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_LIBGD_HEADER
#define RAVL_LIBGD_HEADER 1
////////////////////////////////////////////////////////////
//! rcsid="$Id: JasperFormat.hh 5240 2005-12-06 17:16:50Z plugger $"
//! file="Ravl/Image/ExternalImageIO/JasperFormat.hh"
//! lib=RavlImgGd
//! author="Warren Moore"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include <gd.h>

namespace RavlImageN
{
  
  //: GdImageC
  // Thin wrapper and utility functions for Gd images
  
  class GdImageC
  {
  public:
    GdImageC(const IntT x, const IntT y);
    //: Constructor
    
    ~GdImageC();
    //: Destructor
    
    void Copy(ImageC<ByteRGBValueC> &image);
    //: Copy and RGB iage into the current image
    
    ImageC<ByteRGBValueC> GetImage();
    //: Get a copy of the image
    
    gdImagePtr Ptr()
    { return m_gdImagePtr; }
    //: Get the data pointer
    
  private:
    gdImagePtr m_gdImagePtr;
  };
  
}

#endif
