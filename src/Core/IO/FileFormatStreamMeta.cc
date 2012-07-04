// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////////
//! lib=RavlIO
//! file="Ravl/Core/IO/FileFormatStream.cc"

#include "Ravl/DP/FileFormatStreamMeta.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/SysLog.hh"

namespace RavlN {

  FileFormatStreamMetaBodyC::FileFormatStreamMetaBodyC()
    : FileFormatBodyC("stream","Standard C++ iostream headed with the class name. ")
  {}
  //: Default constructor.

  const type_info &FileFormatStreamMetaBodyC::ProbeLoad(IStreamC &in,const type_info &obj_type) const  {
    if(!in.good())
      return typeid(void);
    StringC classname;
    if(!ReadString(in,classname))
      return typeid(void);

    FileFormatBaseC ff;
    if(!m_class2format.Lookup(classname,ff))
      return typeid(void);
    return ff.DefaultType();
  }
  //: Is stream in std stream format ?

  const type_info &FileFormatStreamMetaBodyC::ProbeLoad(const StringC &filename,IStreamC &in,const type_info &obj_type) const  {
    //cout << "File Probe '" << filename << "' Looking for:" << TypeName(obj_type) << endl;
    if(filename == "") {
      FileFormatBaseC ff;
      if(!m_class2format.Lookup(TypeName(obj_type),ff))
        return typeid(void);
      return ff.DefaultType();
    }
    return ProbeLoad(in,obj_type); // Check load from stream.
  }

  const type_info &FileFormatStreamMetaBodyC::ProbeSave(const StringC &filename,const type_info &obj_type,bool forceFormat) const {
    // If there's no extention or the extention is 'strm' we can handle it.
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return typeid(void);
    if(forceFormat)
      return ff.DefaultType();
    if(filename.IsEmpty())
      return typeid(void); // Nope.
    if(filename[0] == '@')
      return typeid(void); // Nope.
    StringC ext = Extension(filename);
    if(ext == ""  || ext == "strm" || ext == "txt")
      return ff.DefaultType(); // Yep, can save in format.
    return typeid(void); // Nope.
  }

  //: Create a input port for loading.
  // Will create an Invalid port if not supported.
  DPIPortBaseC FileFormatStreamMetaBodyC::CreateInput(IStreamC &in,const type_info &obj_type) const {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPIPortBaseC();
    return ff.CreateInput(in,obj_type);
  }

  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  DPOPortBaseC FileFormatStreamMetaBodyC::CreateOutput(OStreamC &out,const type_info &obj_type) const {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPOPortBaseC();
    return ff.CreateOutput(out,obj_type);
  }

  //: Get prefered IO type.
  const std::type_info &FileFormatStreamMetaBodyC::DefaultType() const
  { return typeid(void); }

  //: Just to make it clear its a streamable format.
  bool FileFormatStreamMetaBodyC::IsStream() const
  { return true; }

  //: Register format
  bool FileFormatStreamMetaBodyC::RegisterFormat(FileFormatBaseC &fileformat) {
    RavlAssert(fileformat.DefaultType() != typeid(void));
    FileFormatBaseC &entry = m_class2format[RavlN::TypeName(fileformat.DefaultType())];
    if(entry.IsValid()) {
      RavlError("File stream format for '%s' already registered.",RavlN::TypeName(DefaultType()));
      return false;
    }
    entry = fileformat;
    return true;
  }


  FileFormatStreamMetaC &DefaultFormatStreamMeta() {
    static FileFormatStreamMetaC ffbsm;
    return ffbsm;
  }

  //: Register file format.

  bool RegisterFormatStreamMeta(FileFormatBaseC &fileformat) {
    return DefaultFormatStreamMeta().RegisterFormat(fileformat);
  }

  //: Register file format.

  void IncludeFileFormatStreamMeta()
  {  }
  
}
