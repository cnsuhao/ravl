// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLRGBFORMAT_HEADER
#define RAVLRGBFORMAT_HEADER 1
////////////////////////////////////////////////////////////
//! author="Charles Galambos"
//! docentry="Image.Image IO"
//! date="29/10/98"
//! rcsid="$Id$"
//! file="amma/Image/VidIO/rgbFormat.hh"
//! example=exImgSeq.cc
//! lib=RavlImage


#include "Ravl/DP/FileFormat.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"

namespace RavlN {

  //! userlevel=Develop
  //: RGB File format information.
  
  class FileFormatRGBBodyC : public FileFormatBodyC {
  public:
    FileFormatRGBBodyC(const StringC &vName,const Index2dC &vSize);
    //: Constructor.
    
    virtual const type_info &ProbeLoad(IStreamC &in,const type_info &obj_type) const;
    //: Is stream in std stream format ?
    
    virtual const type_info &ProbeLoad(const StringC &filename,IStreamC &in,const type_info &obj_type) const;
    //: Probe for load.
    
    virtual const type_info &ProbeSave(const StringC &filename,const type_info &obj_type,bool forceFormat) const;
    //: Probe for Save.
    
    virtual DPIPortBaseC CreateInput(IStreamC &in,const type_info &obj_type) const;
    //: Create a input port for loading.
    // This creates a plain RGB stream.
    // Will create an Invalid port if not supported.
    
    virtual DPOPortBaseC CreateOutput(OStreamC &out,const type_info &obj_type) const;
    //: Create a output port for saving.
    // This creates a plain RGB stream.
    // Will create an Invalid port if not supported.
    
    virtual DPIPortBaseC CreateInput(const StringC &filename,const type_info &obj_type) const;
    //: Create a input port for loading from file 'filename'.
    // Will create an Invalid port if not supported. <p>
    
    virtual DPOPortBaseC CreateOutput(const StringC &filename,const type_info &obj_type) const;
    //: Create a output port for saving to file 'filename'..
    // Will create an Invalid port if not supported. <p>
    
    virtual const type_info &DefaultType() const; 
    //: Get prefered IO type.
    
    virtual IntT Priority() const { return 0; }
    //: Find the priority of the format. the higher the better.
    // Default is zero, use this if nothing better.
    
    virtual bool IsStream() const { return true; }
    //: Test if format is a fully streamable.
    // i.e. check if you can read/write more than object object.
    
  protected:
    StringC vName;  // Variant name.
    Index2dC vSize; // Variant image size.
  };
  
  /////////////////////////////
  //! userlevel=Advanced
  //: Create an instance of a RGB File Format.
  
  class FileFormatRGBC : public FileFormatC<ImageC<ByteRGBValueC> > {
  public:
    FileFormatRGBC(const StringC &vName,const Index2dC &vSize)
      : FileFormatC<ImageC<ByteRGBValueC> >(*new FileFormatRGBBodyC(vName,vSize))
      {}
  };
}
  
#endif
