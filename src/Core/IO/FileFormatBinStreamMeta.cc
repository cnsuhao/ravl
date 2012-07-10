// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////////
//! lib=RavlIO
//! file="Ravl/Core/IO/FileFormatBinStream.cc"

#include "Ravl/DP/FileFormatBinStreamMeta.hh"
#include "Ravl/DP/BinFileIO.hh"
#include "Ravl/SysLog.hh"

namespace RavlN {
  FileFormatBinStreamMetaBodyC::FileFormatBinStreamMetaBodyC(bool pubList)
    : FileFormatBodyC("abs","RAVL binary stream. ",pubList)
  {}
  //: Constructor.

  FileFormatBinStreamMetaBodyC::FileFormatBinStreamMetaBodyC(const StringC &formatId,
                                                             const StringC &formatDescriptor,
                                                             bool pubList)
    : FileFormatBodyC(formatId,formatDescriptor,pubList)
  {}
  //: Constructor with full format info.

  const std::type_info &FileFormatBinStreamMetaBodyC::ProbeLoad(IStreamC &in,const type_info &/*obj_type*/) const
  {
    if(!in.good())
      return typeid(void);
    BinIStreamC bin(in);
    streampos mark = bin.Tell();
    UInt16T id;
    // Check magic number.
    bin >> id;
    switch(id)
    {
      case RavlN::RAVLBinaryID64:
        bin.SetCompatibilityMode32Bit(false);
        break;
      case RavlN::RAVLInvBinaryID64:
        bin.SetCompatibilityMode32Bit(false);
        bin.UseNativeEndian(!bin.NativeEndian());
        break;
      case RavlN::RAVLBinaryID32:
        bin.SetCompatibilityMode32Bit(true);
        break;
      case RavlN::RAVLInvBinaryID32:
        bin.SetCompatibilityMode32Bit(true);
        bin.UseNativeEndian(!bin.NativeEndian());
        break;
      case RavlN::RAVLInvBinaryID:
        bin.UseNativeEndian(!bin.NativeEndian());
        // Fall through
      case RavlN::RAVLBinaryID:
        // Use what every default 32/64 bit mode is set in the stream.
        break;
      default:
        // Unknown format string.
        bin.Seek(mark);
        return typeid(void);
    }
    // Check class name.
    StringC classname;
    bin >> classname;
    //cout << "Stream Probe: '" << classname << "' Looking for: '" << TypeName(typeid(DataT)) << "'\n";
    bin.Seek(mark);
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(classname,ff))
      return typeid(void);
    return ff.DefaultType();
  }
  //: Is stream in std stream format ?

  const std::type_info &FileFormatBinStreamMetaBodyC::ProbeLoad(const StringC &filename,IStreamC &in,const type_info &obj_type) const
  {
    //cout << "File Probe '" << filename << "' Looking for:" << TypeName(obj_type) << endl;
    if(filename == "") {
      // Pure stream ?
      FileFormatBaseC ff;
      if(!m_class2format.Lookup(TypeName(obj_type),ff))
        return typeid(void);
      return ff.DefaultType();
    }

    return ProbeLoad(in,obj_type); // Check load from stream.
  }

  const std::type_info &FileFormatBinStreamMetaBodyC::ProbeSave(const StringC &filename,const type_info &obj_type,bool forceFormat) const
  {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return typeid(void);
    if(forceFormat) {
      return ff.DefaultType();
    }
    StringC ext = Extension(filename);
    // If there's no extension or the extension is 'abs' we can handle it.
    // abs = RAVL Binary Stream.
    if(filename.IsEmpty())
      return typeid(void); // Nope.
    if(filename[0] == '@')
      return typeid(void); // Nope.
    if(ext == ""  || ext == "abs" || ext == "bin" || m_ext.IsMember(ext))
      return ff.DefaultType(); // Yep, can save in format.
    return typeid(void); // Nope.
  }

  DPIPortBaseC FileFormatBinStreamMetaBodyC::CreateInput(IStreamC &in,const std::type_info &obj_type) const {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPIPortBaseC();
    return ff.CreateInput(in,obj_type);
  }
  //: Create a input port for loading.
  // Will create an Invalid port if not supported.

  DPOPortBaseC FileFormatBinStreamMetaBodyC::CreateOutput(OStreamC &out,const std::type_info &obj_type) const {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPOPortBaseC();
    return ff.CreateOutput(out,obj_type);
  }
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.


  DPIPortBaseC FileFormatBinStreamMetaBodyC::CreateInput(const StringC &filename,const std::type_info &obj_type) const  {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPIPortBaseC();
    return ff.CreateInput(filename,obj_type);
  }

  //: Create a input port for loading.
  // Will create an Invalid port if not supported.

  DPOPortBaseC FileFormatBinStreamMetaBodyC::CreateOutput(const StringC &filename,const std::type_info &obj_type) const {
    FileFormatBaseC ff;
    if(!m_class2format.Lookup(TypeName(obj_type),ff))
      return DPOPortBaseC();
    return ff.CreateOutput(filename,obj_type);
  }
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.

  const std::type_info &FileFormatBinStreamMetaBodyC::DefaultType() const
  { return typeid(void); }
  //: Get preferred IO type.

  bool FileFormatBinStreamMetaBodyC::IsStream() const
  { return true; }
  //: Just to make it clear its a streamable format.

  //: Register format
  bool FileFormatBinStreamMetaBodyC::RegisterFormat(FileFormatBaseC &fileformat) {
    RavlAssert(fileformat.DefaultType() != typeid(void));
    FileFormatBaseC &entry = m_class2format[RavlN::TypeName(fileformat.DefaultType())];
    if(entry.IsValid()) {
      RavlError("File abs format for '%s' already registered.",RavlN::TypeName(DefaultType()));
      return false;
    }
    entry = fileformat;
    m_ext += fileformat.Name();
    return true;
  }


  FileFormatBinStreamMetaC &DefaultFormatBinStreamMeta() {
    static FileFormatBinStreamMetaC ffbsm("abs","RAVL binary file formats");
    return ffbsm;
  }

  //: Register file format.

  bool RegisterFormatBinStreamMeta(FileFormatBaseC &fileformat) {
    return DefaultFormatBinStreamMeta().RegisterFormat(fileformat);
  }

  
  void InitFileFormatBinStreamMeta()
  {}
  
}
