// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id$"
//! docentry="Ravl.Contrib.Video IO"
//! lib=RavlImgIO1394dc
//! file="Ravl/Contrib/1394dc/1394dcFormat.cc"
//! author="Charles Galambos"

#include "Ravl/Image/Lib1394dcFormat.hh"
#include "Ravl/Image/ImgIO1394dc.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/Image/RealYUVValue.hh"
#include "Ravl/Image/RealRGBValue.hh"

#define DPDEBUG 1
#include "Ravl/TypeName.hh"
#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  void Init1394dcFormat()
  {}
  
  // 1394dc ////////////////////////////////////////////////////////////////
  
  //: Constructor.
  FileFormat1394dcBodyC::FileFormat1394dcBodyC(const StringC &nvName)
    : FileFormatBodyC(nvName,StringC("Lib1394dc firewire camera driver. (@IIDC)")),
      vName(nvName)
  {}
  
  
  const type_info &FileFormat1394dcBodyC::ProbeLoad(IStreamC &in,const type_info &obj_type) const
  { return typeid(void); }
  
  
  const type_info &
  FileFormat1394dcBodyC::ProbeLoad(const StringC &filename,IStreamC &in,const type_info &obj_type) const { 
    if(filename.length() == 0)
      return typeid(void);
    if(filename[0] != '@')
      return typeid(void);
    StringC device = ExtractDevice(filename);
    StringC file = ExtractParams(filename);
    int channel = 0;
    int pn = file.index('#');
    if(pn >= 0) { // Got a channel number ?
      channel = file.after(pn).IntValue();
      file = file.before(pn);
    }
    ONDEBUG(cerr << "FileFormat1394dcBodyC::ProbeLoad(), Checking file type." << TypeName(obj_type.name()) << " Device='" << device <<"'\n");
    if(device != "IIDC")
      return typeid(void);
    
    enum { IMG_RGB, IMG_YUV, IMG_YUV422, IMG_GREY } imgtype = IMG_GREY;
    
    // Some huristics to select the best format to capture date from the
    // card in.   If in doubt get YUV as thats what most video is in anyway.
#if 0    
    if(obj_type == typeid(ImageC<ByteRGBValueC>))
      imgtype = IMG_RGB;
    if(obj_type == typeid(ImageC<RealRGBValueC>))
      imgtype = IMG_RGB;
    else if(obj_type == typeid(ImageC<ByteYUVValueC>))
      imgtype = IMG_YUV;
    else if(obj_type == typeid(ImageC<RealYUVValueC>))
      imgtype = IMG_YUV;
    else if(obj_type == typeid(ImageC<ByteYUV422ValueC>))
      imgtype = IMG_YUV422;
    else if(obj_type == typeid(ImageC<RealT>))
      imgtype = IMG_GREY;
    else if(obj_type == typeid(ImageC<IntT>))
      imgtype = IMG_GREY;
    else if(obj_type == typeid(ImageC<UIntT>))
      imgtype = IMG_GREY;
    //else
#endif 
    if(obj_type == typeid(ImageC<ByteT>))
      imgtype = IMG_GREY;
    
    switch(imgtype) {
    case IMG_GREY: return typeid(ImageC<ByteT>);
    case IMG_RGB: //return typeid(ImageC<ByteRGBValueC>);
    case IMG_YUV: //return typeid(ImageC<ByteYUVValueC>);
    case IMG_YUV422: //return typeid(ImageC<ByteYUV422ValueC>);
      cerr << "Unsupported pixel type. \n";
      return typeid(void);
    }
    return typeid(ImageC<ByteT>);
  }
  
  const type_info &
  FileFormat1394dcBodyC::ProbeSave(const StringC &nfilename,const type_info &obj_type,bool forceFormat) const
  { return typeid(void); }
  
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  
  DPIPortBaseC FileFormat1394dcBodyC::CreateInput(IStreamC &in,const type_info &obj_type) const
  { return DPIPortBaseC(); }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC FileFormat1394dcBodyC::CreateOutput(OStreamC &out,const type_info &obj_type) const 
  { return DPOPortBaseC(); }

  //: Create a input port for loading from file 'filename'.
  // Will create an Invalid port if not supported. <p>
  
  DPIPortBaseC FileFormat1394dcBodyC::CreateInput(const StringC &filename,const type_info &obj_type) const
  {
    ONDEBUG(cerr << "FileFormat1394dcBodyC::CreateInput(const StringC &,const type_info &), Called. \n");
    if(filename[0] != '@')
      return DPIPortBaseC();
    StringC fn = ExtractParams(filename);
    StringC dev = ExtractDevice(filename);
    int channel = 0;
    int pn = fn.index('#');
    if(pn >= 0) { // Got a channel number ?
      channel = fn.after(pn).IntValue();
      fn = fn.before(pn);
    }
    //bool half = false;
    //if(dev == "IIDC")
    //  half = true; // Attempt to get images halfed along each dimention.
    if(fn == "")
      fn = "/dev/raw1394";
#if 0
    if(obj_type == typeid(ImageC<ByteYUVValueC>))
      return DPIImage1394dcC<ByteYUVValueC>(fn,half,channel);
    if(obj_type == typeid(ImageC<ByteYUV422ValueC>))
      return DPIImage1394dcC<ByteYUV422ValueC>(fn,half,channel);
    if(obj_type == typeid(ImageC<ByteRGBValueC>))
      return DPIImage1394dcC<ByteRGBValueC>(fn,half,channel);
#endif
    if(obj_type == typeid(ImageC<ByteT>))
      return DPIImage1394dcC<ByteT>(fn,channel); //,half,channel
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving to file 'filename'..
  // Will create an Invalid port if not supported. <p>
  
  DPOPortBaseC FileFormat1394dcBodyC::CreateOutput(const StringC &filename,const type_info &obj_type) const
  { return DPOPortBaseC(); }
  
  //: Get prefered IO type.
  
  const type_info &FileFormat1394dcBodyC::DefaultType() const 
  { return typeid(ImageC<ByteT>); }
  
  // Some common cif formats.
  
  FileFormat1394dcC RegisterFileFormat1394dc("1394dc");
}
