// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id$"
//! docentry="Ravl.Contrib.Image IO"
//! lib=RavlImgIOV4L
//! file="Contrib/V4L/V4LFormat.cc"

#include "Ravl/Image/V4LFormat.hh"
#include "Ravl/Image/ImgIOV4L.hh"
#include "Ravl/TypeName.hh"

#define DPDEBUG 0

#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  void InitV4LFormat()
  {}
  
  // V4L ////////////////////////////////////////////////////////////////
  
  //: Constructor.
  FileFormatV4LBodyC::FileFormatV4LBodyC(const StringC &nvName)
    : FileFormatBodyC(nvName,StringC("V4L driver.")),
      vName(nvName)
  {}
  
  
  const type_info &
  FileFormatV4LBodyC::ProbeLoad(IStreamC &in,const type_info &obj_type) const
  { return typeid(void); }
  
  const type_info &
  FileFormatV4LBodyC::ProbeLoad(const StringC &filename,IStreamC &in,const type_info &obj_type) const { 
    if(filename.length() == 0)
      return typeid(void);
    if(filename[0] != '@')
      return typeid(void);
    StringC device = ExtractDevice(filename);
    if(device != "V4L" && device != "V4LH")
      return typeid(void);
#if 0
    if(obj_type == typeid(ImageC<ByteRGBValueC>))
      return typeid(ImageC<ByteRGBValueC>); 
#endif
    return typeid(ImageC<ByteYUVValueC>);
  }
  
  const type_info &
  FileFormatV4LBodyC::ProbeSave(const StringC &nfilename,const type_info &obj_type,bool forceFormat) const
  { return typeid(void); }
  
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  
  DPIPortBaseC FileFormatV4LBodyC::CreateInput(IStreamC &in,const type_info &obj_type) const
  { return DPIPortBaseC(); }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC FileFormatV4LBodyC::CreateOutput(OStreamC &out,const type_info &obj_type) const 
  { return DPOPortBaseC(); }

  //: Create a input port for loading from file 'filename'.
  // Will create an Invalid port if not supported. <p>
  
  DPIPortBaseC FileFormatV4LBodyC::CreateInput(const StringC &filename,const type_info &obj_type) const
  {
    ONDEBUG(cerr << "FileFormatV4LBodyC::CreateInput(const StringC &,const type_info &), Called. \n");
    if(filename[0] != '@')
    return DPIPortBaseC();
    StringC fn = ExtractParams(filename);
    StringC dev = ExtractDevice(filename);
    bool half = false;
    if(dev == "V4LH")
      half = true; // Attempt to get images halfed along each dimention.
    if(fn == "")
      fn = "/dev/video0";
    if(obj_type == typeid(ImageC<ByteYUVValueC>))
      return DPIImageV4LC<ByteYUVValueC>(fn,half);
#if 0
    if(obj_type == typeid(ImageC<ByteRGBValueC>))
      return DPIImageV4LC<ByteRGBValueC>(fn,half);
#endif
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving to file 'filename'..
  // Will create an Invalid port if not supported. <p>
  
  DPOPortBaseC FileFormatV4LBodyC::CreateOutput(const StringC &filename,const type_info &obj_type) const
  { return DPOPortBaseC(); }
  
  //: Get prefered IO type.
  
  const type_info &FileFormatV4LBodyC::DefaultType() const 
  { return typeid(ImageC<ByteYUVValueC>); }
  
  // Some common cif formats.
  
  FileFormatV4LC RegisterFileFormatV4L("v4l");
}
