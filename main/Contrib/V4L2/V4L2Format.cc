// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlIOV4L2
//! author = "Warren Moore"

#include "Ravl/Image/V4L2Format.hh"
//#include "Ravl/Image/IOV4L2.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/Image/RealYUVValue.hh"
#include "Ravl/Image/RealRGBValue.hh"

#define DPDEBUG 0
#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN
{
  
  void InitV4L2Format()
  {}
  
  FileFormatV4L2BodyC::FileFormatV4L2BodyC() :
    FileFormatBodyC("v4l2", "V4L2 file input. ")
  {}
  
  const type_info& FileFormatV4L2BodyC::ProbeLoad(IStreamC &in, const type_info &obj_type) const
  {
    return typeid(void); 
  }
  
  const type_info& FileFormatV4L2BodyC::ProbeLoad(const StringC &filename, IStreamC &in, const type_info &obj_type) const
  {
    return typeid(void);
  }
  
  const type_info& FileFormatV4L2BodyC::ProbeSave(const StringC &filename, const type_info &obj_type, bool forceFormat ) const
  {
    return typeid(void);   
  }
  
  DPIPortBaseC FileFormatV4L2BodyC::CreateInput(IStreamC &in, const type_info &obj_type) const
  { 
    return DPIPortBaseC();
  }
  
  DPOPortBaseC FileFormatV4L2BodyC::CreateOutput(OStreamC &out, const type_info &obj_type) const
  {
    return DPOPortBaseC();  
  }
  
  DPIPortBaseC FileFormatV4L2BodyC::CreateInput(const StringC &filename, const type_info &obj_type) const
  {
    return DPIPortBaseC();
  }
  
  DPOPortBaseC FileFormatV4L2BodyC::CreateOutput(const StringC &filename, const type_info &obj_type) const
  {
    return DPOPortBaseC();  
  }
  
  const type_info& FileFormatV4L2BodyC::DefaultType() const
  { 
    return typeid(ImageC<ByteRGBValueC>); 
  }
  
  static FileFormatV4L2C Init;  
}
