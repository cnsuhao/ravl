// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImageIO

#include "Ravl/3D/FormatTriFile.hh"
#include "Ravl/3D/TriFileIO.hh"
#include "Ravl/3D/TriSet.hh"
#include "Ravl/TypeName.hh"

#define DPDEBUG 0
#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace Ravl3DN {

  void InitTriFormat() {
  }
    
  // Tri ////////////////////////////////////////////////////////////////
  
  //: Constructor.
  
  FileFormatTriBodyC::FileFormatTriBodyC()
    : FileFormatBodyC("tri","3D Mesh.")
  {}
  
  //: Is stream in std stream format ?
  
  const type_info &
  FileFormatTriBodyC::ProbeLoad(IStreamC &in,const type_info &obj_type) const {
    return typeid(TriSetC);
  }

  const type_info & FileFormatTriBodyC::ProbeLoad(const StringC &nfilename,IStreamC &in,const type_info &obj_type) const {
    //StringC filename(nfilename);
    //  if(obj_type != typeid(TriSetC))
    //    return false; // Can only deal with rgb at the moment.
    // For Load, use stream probe its more reliable than extentions.
    return ProbeLoad(in,obj_type);
  }
  
  const type_info & FileFormatTriBodyC::ProbeSave(const StringC &filename,const type_info &obj_type,bool forceFormat ) const {
    cerr << "FileFormatTriBodyC::ProbeSave().. \n";
    if(forceFormat)
      return typeid(TriSetC);
    if(!Extension(filename) == StringC(".tri") && filename != "-")
      return typeid(void);
    return typeid(TriSetC);
  }
  
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  
  DPIPortBaseC FileFormatTriBodyC::CreateInput(IStreamC &in,const type_info &obj_type) const {
    if(obj_type == typeid(TriSetC))
      return DPITriFileC(in);
    return DPIPortBaseC();
  }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC FileFormatTriBodyC::CreateOutput(OStreamC &out,const type_info &obj_type) const  {
    if(obj_type == typeid(TriSetC))
      return  DPOTriFileC(out);
    return DPOPortBaseC();
  }
  
  //: Get prefered IO type.
  
  const type_info &FileFormatTriBodyC::DefaultType() const 
  { return typeid(TriSetC); }
  
  
  //////////////////////////////////////////////////////////////////
  
  FileFormatTriC RegisterFileFormatTri;
  
}
