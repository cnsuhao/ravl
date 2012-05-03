// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001-12, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id: dvsFormat.cc 342 2001-08-28 13:33:14Z simone $"
//! lib=RavlRawVidIO
//! file="Ravl/DVSFileFormat/dvsFormat.cc"

#include "Ravl/Image/dvsFormat.hh"
#include "Ravl/Image/ImgIOdvsypbpr.hh"
#include "Ravl/Image/ImgIOdvsrgb.hh"
#include "Ravl/TypeName.hh"
#include <ctype.h>
#include "Ravl/Image/NewGrabfileReader.hh"
#include "Ravl/Image/NewGrabfileWriter.hh"
#include "Ravl/Stream.hh"
#include "Ravl/StrStream.hh"


using namespace RavlImageN;
using namespace RavlN;

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  void InitDVSYUV422Format() 
  {
  }
  
  // DVSYUV422 ////////////////////////////////////////////////////////////////
  
  FileFormatDVSYPbPr422BodyC::FileFormatDVSYPbPr422BodyC(const StringC &nvName,const Index2dC &nvSize)
    : FileFormatBodyC(nvName,StringC("YUV422 images Size:") + StringC(nvSize.Row().V()) + " " + StringC(nvSize.Col().V())),
      vName(nvName),
      vSize(nvSize)
  {
  file_writer_pointer = 0;
  file_reader_pointer = 0;
  }
  //: Constructor.
  
  //: Is stream in std stream format ?
  // No way to tell !!
  // Just say no ??
  
  const type_info &
  FileFormatDVSYPbPr422BodyC::ProbeLoad(IStreamC &in,const type_info &obj_type) const { 
    file_reader_pointer = NewGrabfileReader(in.Name(), false);
    if(file_reader_pointer->Open(in.Name().chars())) { //, mode)) {
       switch(IdToColourMode(file_reader_pointer->ColourMode())) {
          case YUV422: {
             return typeid(ImageC<FloatYPbPr422BT709ValueC>);
          }
          case RGB_RGB: {
             return typeid(ImageC<ByteRGBValueC>);
          }
          default: {
             RavlIssueError("Unknown Colour Mode in FileFormatYPbPrBodyC::ProbeLoad.") ;
          }
        }
    }
    //else {
    //Got this far then unable to open grab file so issue error.
       RavlIssueError("Could not open grab file.") ;
       return typeid(void);
    //}
  }
  
  const type_info &
  FileFormatDVSYPbPr422BodyC::ProbeLoad(const StringC &nfilename,IStreamC &in,const type_info &obj_type) const {
    StringC suffix = Extension(nfilename);
    ONDEBUG(cerr << "FileFormatDVSYPbPr422BodyC::ProbeLoad() [" << vName << "] Called. Filename:'"<<nfilename <<" Ext:'" << suffix << "'  LoadType:'" << TypeName(obj_type) << "'\n");
    if (suffix != vName) {     
       return typeid(void);
    }
    else {
       return ProbeLoad(in,obj_type);
    }
  }
  
  const type_info &
  FileFormatDVSYPbPr422BodyC::ProbeSave(const StringC &nfilename,const type_info &obj_type,bool forceFormat) const {
    if(!forceFormat) {
      StringC suffix = Extension(nfilename);
      ONDEBUG(cerr << "FileFormatDVSYPbPr422BodyC::ProbeSave() [" << vName << "] Called. Filename:'"<<nfilename <<" Ext:'" << suffix << "'  LoadType:'" << TypeName(obj_type) << "'\n");
      if (suffix != vName) {     
	return typeid(void);
      }
    }
       if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
          cmode = YUV422;
	  bformat = BITS_10_DVS;
	  vmode = SMPTE274_25I;
       }
       if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
          cmode = RGB_RGB;
	  bformat = BITS_8;
	  vmode = PAL;
       }
       //: Setup the card
   
	file_writer_pointer = NewGrabfileWriter(1);
    if(file_writer_pointer->Open(nfilename,IdToVideoMode(vmode),IdToByteFormat(bformat),IdToColourMode(cmode),1245184,0)) {
       //Check for YUV422 or RGB in header then return appropriate typeid.
          if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
             return typeid(ImageC<FloatYPbPr422BT709ValueC>);
          }
	  else if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
             return typeid(ImageC<ByteRGBValueC>);
          }
	  else {
             RavlIssueError("Unknown Colour Mode in FileFormatYPbPrBodyC::ProbeLoad.") ;
	     return typeid(void);
          }
        }
	RavlIssueError("Could not open output grabfile");
        return typeid(void);
    }
  
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  
  DPIPortBaseC FileFormatDVSYPbPr422BodyC::CreateInput(IStreamC &in,const type_info &obj_type) const {
    ONDEBUG(cerr << "FileFormatDVSYPbPr422BodyC::CreateInput(IStreamC &,const type_info &), Called. \n");
    if(!in.good()) {
      return DPIPortBaseC();
    }
    if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
      return DPIImageDVSYPbPr422C(*file_reader_pointer,in,vSize);
    }
    if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
      return DPIImageDVSRGBC(*file_reader_pointer,in,Index2dC(576,720));
    }
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC FileFormatDVSYPbPr422BodyC::CreateOutput(OStreamC &out,const type_info &obj_type) const  {
    /*if(!out.good()) {
      return DPOPortBaseC();
    }*/
    if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
      return DPOImageDVSYPbPr422C(*file_writer_pointer,out,vSize);
    }
    if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
      return DPOImageDVSRGBC(*file_writer_pointer,out,vSize);
    }
    return DPOPortBaseC();
  }
  
  //: Create a input port for loading from file 'filename'.
  // Will create an Invalid port if not supported. <p>
  
  DPIPortBaseC FileFormatDVSYPbPr422BodyC::CreateInput(const StringC &filename,const type_info &obj_type) const {
    ONDEBUG(cerr << "FileFormatDVSYPbPr422BodyC::CreateInput(const StringC &,const type_info &), Called. \n");
    if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
      IStreamC strm(filename);
      if(!strm) {
	return DPIPortBaseC();
      }
      return DPIImageDVSYPbPr422C(*file_reader_pointer,strm,vSize);
    }
    if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
      IStreamC strm(filename);
      if(!strm) {
	return DPIPortBaseC();
      }
      return DPIImageDVSRGBC(*file_reader_pointer,strm,vSize);
    }
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving to file 'filename'..
  // Will create an Invalid port if not supported. <p>
  
  DPOPortBaseC FileFormatDVSYPbPr422BodyC::CreateOutput(const StringC &filename,const type_info &obj_type) const {
    if(obj_type == typeid(ImageC<FloatYPbPr422BT709ValueC>)) {
      OStreamC strm(filename);
      return DPOImageDVSYPbPr422C(*file_writer_pointer,strm,vSize);
    }
    if(obj_type == typeid(ImageC<ByteRGBValueC>)) {
      OStreamC strm(filename);
      return DPOImageDVSRGBC(*file_writer_pointer,strm,vSize);
    }
    return DPOPortBaseC();
  }
  

  //Destructor.
  FileFormatDVSYPbPr422BodyC::~FileFormatDVSYPbPr422BodyC() {
     if(file_writer_pointer != NULL) {
        file_writer_pointer->Close();
     }
     if(file_reader_pointer != NULL) {
        file_reader_pointer->Close();
     }
  }

  //: Get prefered IO type.
  
  const type_info &FileFormatDVSYPbPr422BodyC::DefaultType() const 
  { return typeid(ImageC<FloatYPbPr422BT709ValueC>); }
  
  
  // Some common dvs file formats.
  //FileFormatDVSRGBC RegisterFileFormatDVSRGB ("grab",Index2dC(576,720));
  FileFormatDVSYPbPr422C RegisterFileFormatDVSYPbPr422  ("grab"  ,Index2dC(1080,1920));
}
