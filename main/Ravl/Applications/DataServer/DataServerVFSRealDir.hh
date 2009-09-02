// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_DATASERVERVFSREALDIR_HEADER
#define RAVL_DATASERVERVFSREALDIR_HEADER 1

#include "Ravl/DataServer/DataServerVFSNode.hh"
#include "Ravl/DataServer/DataServerVFSRealFile.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/DP/TypeInfo.hh"
#include "Ravl/DP/SPortShare.hh"
#include "Ravl/RCWrap.hh"

namespace RavlN {
  
  //! userlevel=Develop
  //: Handle file's or directories in a real filesystem.
  
  class DataServerVFSRealDirBodyC
    : public DataServerVFSNodeBodyC
  {
  public:
    DataServerVFSRealDirBodyC(const StringC &vname,const StringC &nRealDirname,bool canWrite,bool canCreate_);
    //: Constructor.
    
    ~DataServerVFSRealDirBodyC();
    //: Destructor.
    
    virtual bool Configure(const ConfigFileC &config);
    //: Configure node with given setup.
    
    const FilenameC &RealDirname() const
    { return realDirname; }
    //: Real filename.
    
    virtual bool OpenIPort(DListC<StringC> &remainingPath,const StringC &dataType,NetISPortServerBaseC &port);
    //: Open input port.
    
    virtual bool OpenOPort(DListC<StringC> &remainingPath,const StringC &dataType,NetOSPortServerBaseC &port);
    //: Open output port.
    
    virtual bool Delete(DListC<StringC>& remainingPath);
    //: Delete the physical media of the target path within the node.
    //!param: remainingPath - List of strings containing the path elements to the target within the node.
    //!return: True if successfully deleted.

    virtual bool QueryNodeSpace(const StringC& remainingPath, Int64T& total, Int64T& used, Int64T& available);
    //: Query physical media details for the target path within the node.
    //!param: remainingPath - List of strings containing the path elements to the target within the node.
    //!param: total - Returns the space allocated for the partition containing the target path in bytes (both free and used). -1 if not applicable.
    //!param: used - Returns the space used on the partition containing the target path in bytes. -1 if not applicable.
    //!param: available - Returns the space available on the partition containing the target node in bytes. -1 if not applicable.
    //!return: True if the query executed successfully.

  protected:
    bool OpenVFSFile(DListC<StringC> &remainingPath,DataServerVFSRealFileC &rfile,bool forWrite = false);
    //: Open VFS file.
    
    MutexC access; // Access control for object.
    HashC<StringC,DataServerVFSRealFileC> name2file;
    
    FilenameC realDirname;

    bool canCreate;
  };
  
  //! userlevel=Normal
  //: Handle file's or directories in a real filesystem. 
  //!cwiz:author
  
  class DataServerVFSRealDirC
    : public DataServerVFSNodeC
  {
  public:
    DataServerVFSRealDirC(const StringC & vname,const StringC & nRealDirname,bool canWrite = false, bool canCreate = false)
      : DataServerVFSNodeC(*new DataServerVFSRealDirBodyC(vname,nRealDirname,canWrite,canCreate))
    {}
    //: Constructor. 
    //!cwiz:author
    
    const FilenameC & RealDirname() const
    { return Body().RealDirname(); }
    //: Real filename. 
    //!cwiz:author
    
  protected:
    DataServerVFSRealDirC(DataServerVFSRealDirBodyC &bod)
     : DataServerVFSNodeC(bod)
    {}
    //: Body constructor. 
    
    DataServerVFSRealDirBodyC& Body()
    { return static_cast<DataServerVFSRealDirBodyC &>(DataServerVFSNodeC::Body()); }
    //: Body Access. 
    
    const DataServerVFSRealDirBodyC& Body() const
    { return static_cast<const DataServerVFSRealDirBodyC &>(DataServerVFSNodeC::Body()); }
    //: Body Access. 
    
  };
}


#endif
