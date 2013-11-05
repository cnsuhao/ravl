// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlImageIO
//! file="Ravl/Image/ImageIO/YUVFormat.cc"

#include "Ravl/BinStream.hh"
#include "Ravl/Image/YUVFormat.hh"
#include "Ravl/Image/ImgIOYUV.hh"
#include "Ravl/TypeName.hh"
#include <ctype.h>

#define DPDEBUG 0

#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  void InitYUVFormat() {
  }
  
  //: Constructor.
  
  FileFormatYUVBodyC::FileFormatYUVBodyC()
    : FileFormatBodyC("yuv","YUV 422 images.")
  {}
  
  //: Is stream in std stream format ?
  
  const std::type_info &
  FileFormatYUVBodyC::ProbeLoad(IStreamC &in,const std::type_info &obj_type) const {
    if(!in.good())
      return typeid(void);
    return typeid(ImageC<ByteYUVValueC>);
  }

  const std::type_info &
  FileFormatYUVBodyC::ProbeLoad(const StringC &nfilename,IStreamC &in,const std::type_info &obj_type) const {
    return ProbeLoad(in,obj_type);
  }
  
  const std::type_info &
  FileFormatYUVBodyC::ProbeSave(const StringC &filename,const std::type_info &obj_type,bool forceFormat ) const {
    if(forceFormat)
      return typeid(ImageC<ByteYUVValueC>);
    if(Extension(filename) != StringC("yuv") && filename != "-")
      return typeid(void);
    return typeid(ImageC<ByteYUVValueC>);
  }
  
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  
  DPIPortBaseC FileFormatYUVBodyC::CreateInput(IStreamC &in,const std::type_info &obj_type) const {
    if(obj_type == typeid(ImageC<ByteYUVValueC>))
      return DPIImageYUVC(in);
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC FileFormatYUVBodyC::CreateOutput(OStreamC &out,const std::type_info &obj_type) const  {
    if(obj_type == typeid(ImageC<ByteYUVValueC>))
      return  DPOImageYUVC(out);
    return DPOPortBaseC();
  }
  
  //: Get prefered IO type.
  
  const std::type_info &FileFormatYUVBodyC::DefaultType() const 
  { return typeid(ImageC<ByteYUVValueC>); }
  
  
  FileFormatYUVC RegisterFileFormatYUV;
}
