// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! author="Charles Galambos"
//! rcsid="$Id$"
//! lib=RavlDV
//! file="Ravl/Contrib/DV/DvFrameConvert.cc"

#include "Ravl/Image/DvDecode.hh"
#include "Ravl/Image/Deinterlace.hh"
#include "Ravl/DP/Converter.hh"
#include "Ravl/Image/DVFrame.hh"

namespace RavlImageN {
  
  void InitDvFrameConvert()
  {}
  
  ImageC<ByteRGBValueC> ConvertDvFrame2RGBImage(const DVFrameC &dvf) {
    DvDecodeC decoder(true);
    return decoder.Apply((ByteT *) &(const_cast<DVFrameC &>(dvf)[0]));
  }
  
  DP_REGISTER_CONVERSION_FT_NAMED(DVFrameC,ImageC<ByteRGBValueC>,ConvertDvFrame2RGBImage,1,"ImageC<ByteRGBValueC> RavlImageN::Convert(const DvFrameC &)");
  
}
