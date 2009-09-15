// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#include "Ravl/DataServer/DataServerVFSRealFile.hh"
#include "Ravl/OS/NetPortManager.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/DP/CacheIStream.hh"
#include "Ravl/DP/SerialisePort.hh"
#include "Ravl/DP/TypeConverter.hh"
#include "Ravl/DP/FileFormatDesc.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  
  //: Constructor.
  
  DataServerVFSRealFileBodyC::DataServerVFSRealFileBodyC(const StringC &vname,const StringC& npath,const StringC &nRealFilename,bool canWrite)
    : DataServerVFSNodeBodyC(vname,npath,canWrite,false),
      cacheSize(0),
      realFilename(nRealFilename),
      canSeek(true),
      multiWrite(false),
      deleteOnClose(false)
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DataServerVFSRealFileBodyC, Called name=" << name << " path=" << path << "\n");
  }
  
  //: Destructor.
  
  DataServerVFSRealFileBodyC::~DataServerVFSRealFileBodyC() {
    MutexLockC lock(access);
    CloseIFileAbstract();
    CloseIFileByte();
    CloseOFile();
    DeleteOnClose();
  }
  
  //: Configure node with given setup.
  
  bool DataServerVFSRealFileBodyC::Configure(const ConfigFileC &config) {
    
    // Lock server while we're setting up.
    // No other threads should be running here, but better safe than sorry.
    MutexLockC lock(access);
    
    // Configure basic attributes.
    DataServerVFSNodeBodyC::Configure(config);
    
    // Setup filename 
    realFilename = config["Filename"];

    // File format
    defaultFileFormat = config["FileFormat"];
    
    // Setup cache.
    StringC cacheSizeStr = config["CacheSize"];
    if(!cacheSizeStr.IsEmpty())
      cacheSize = cacheSizeStr.UIntValue();
    else
      cacheSize = 0;
    
    // Setup multiwrite.
    StringC multiWriteStr = config["MultiWrite"];
    if(multiWriteStr == "1")
      multiWrite = true;
    else
      multiWrite = false;
    
    return true;
  }
  
  //: Open input port.
  
  bool DataServerVFSRealFileBodyC::OpenIPort(DListC<StringC> &remainingPath, const StringC &dataType, NetISPortServerBaseC &port)
  {
    if (!remainingPath.IsEmpty())
    {
      cerr << "DataServerVFSRealFileBodyC::OpenIPort failed to open as path remaining after a valid filename." << endl;
      return false;
    }

    if (oport.IsValid())
    {
      cerr << "DataServerVFSRealFileBodyC::OpenIPort failed to open for reading, as already open for writing." << endl;
      return false;
    }

    // Sort out default type.
    if (!iTypeInfo.IsValid())
    {
      StringC useType;
      if (!defaultDataType.IsEmpty())
        useType = defaultDataType;
      else
        useType = dataType;

      iTypeInfo = TypeInfo(RTypeInfo(useType));
      if (!iTypeInfo.IsValid())
      {
        cerr << "DataServerVFSRealFileBodyC::OpenIPort failed to open as type '" << useType << "' unknown" << endl;
        return false;
      }
    }

    MutexLockC lock(access);

    // Choose the correct input port
    if (iTypeInfo.TypeInfo() == typeid(ByteT) && defaultFileFormat == "bytefile")
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenIPort opening byte file for '" << name << "'" << endl);
      return OpenFileReadByte(dataType, port);
    }

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenIPort opening abstract file for '" << name << "'" << endl);
    return OpenFileReadAbstract(dataType, port);
  }
  
  //: Open output port.
  
  bool DataServerVFSRealFileBodyC::OpenOPort(DListC<StringC> &remainingPath,const StringC &dataType,NetOSPortServerBaseC &port) {
    
    // Check the path is empty.
    if(!remainingPath.IsEmpty()) {
      cerr << "DataServerVFSRealFileBodyC::OpenOPort, ERROR: Path remaining after a valid filename. \n";
      return false;
    }
    
    if (iSPortShareAbstract.IsValid() || iSPortShareByte.IsValid())
    {
      cerr << "DataServerVFSRealFileBodyC::OpenIPort failed to open for writing, as already open for reading." << endl;
      return false;
    }

    // Make sure file is open for writing.
    
    if(!OpenFileWrite(dataType))
      return false;
    
    MutexLockC lock(access);
    if(oport.References() > 1 && !multiWrite) {
      cerr << "DataServerVFSRealFileBodyC::OpenOPort, Port already open for writing. \n";
      return false;
    }
    
    // Still don't know what to do...
    DPOPortBaseC realPort = oTypeInfo.CreateConvFromAbstract(oport);

    // Need to do any type conversion ?
    
    if(dataType != TypeName(oTypeInfo.TypeInfo())) {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenOPort, Building type converter. \n");
      DListC<DPConverterBaseC> convSeq = SystemTypeConverter().FindConversion(RTypeInfo(dataType),realPort.OutputType());
      if(convSeq.IsEmpty()) {
        cerr << "DataServerVFSRealFileBodyC::OpenOPort, Failed to find type conversion. \n";
        return false;
      }
      realPort = FileFormatDescC(convSeq,false).BuildOutputConv(realPort);
    }
    
    // Setup client specific part of server to handle connection

    port = NetOSPortServerBaseC(AttributeCtrlC(realPort),
                                realPort,
                                DPSeekCtrlC(realPort),
                                name);

    DataServerVFSRealFileC me(*this);
    Connect(port.SigConnectionClosed(),me,&DataServerVFSRealFileC::DisconnectOPortClient);

    return true;
  }



  bool DataServerVFSRealFileBodyC::Delete()
  {
    if (!CanWrite())
    {
      cerr << "DataServerVFSRealFileBodyC::Delete trying to delete read-only file '" << name << "'" << endl;
      return false;
    }

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::Delete marking for delete" << endl);
    deleteOnClose = true;
    return true;
  }
  
  //: Open file and setup cache.
  
  bool DataServerVFSRealFileBodyC::OpenFileReadAbstract(const StringC& dataType, NetISPortServerBaseC& netPort)
  {
    if (!iSPortShareAbstract.IsValid())
    {
      DPIPortBaseC iPortBase;
      DPSeekCtrlC seekControl;
      if (!OpenISequenceBase(iPortBase, seekControl, realFilename, defaultFileFormat, iTypeInfo.TypeInfo(), verbose))
      {
        cerr << "DataServerBodyC::OpenFileReadAbstract failed to open stream '" << name << "' of type '" << iTypeInfo.Name() << "'" << endl;
        return false;
      }

      // Convert into an abstact stream.
      DPIPortC<RCWrapAbstractC> iPortAbstract = iTypeInfo.CreateConvToAbstract(iPortBase);

      // Setup raw input sport.
      DPISPortAttachC<RCWrapAbstractC> iSPortAttachAbstract(iPortAbstract, seekControl);

      // Setup inline cache?
      if (cacheSize > 0)
      {
        ONDEBUG(cerr << "DataServerBodyC::OpenFileReadAbstract added cache size=" << cacheSize << endl);
        iSPortAttachAbstract = CacheIStreamC<RCWrapAbstractC>(iSPortAttachAbstract, cacheSize);
      }

      // Setup port share.
      iSPortShareAbstract = DPISPortShareC<RCWrapAbstractC>(iSPortAttachAbstract);

      // Set trigger to let us know when people stop using this file.
      iSPortShareAbstract.TriggerCountZero() = TriggerR(*this, &DataServerVFSRealFileBodyC::ZeroIPortClientsAbstract);
    }
    RavlAssert(iSPortShareAbstract.IsValid());

    DPISPortC<RCWrapAbstractC> iSPortAbstract = iSPortShareAbstract.Port();
    DPIPortBaseC iPortBase = iTypeInfo.CreateConvFromAbstract(iSPortAbstract);

    if (!AddTypeConversion(dataType, iPortBase))
      return false;

    netPort = NetISPortServerBaseC(AttributeCtrlC(iPortBase),
                                   iPortBase,
                                   iSPortAbstract,
                                   name);

    return true;
  }



  bool DataServerVFSRealFileBodyC::OpenFileReadByte(const StringC& dataType, NetISPortServerBaseC& netPort)
  {
    DPIPortBaseC iPortBase;
    DPISPortC<ByteT> iSPortByte;
    if (!iSPortShareByte.IsValid())
    {
      DPSeekCtrlC seekControl;
      if (!OpenISequenceBase(iPortBase, seekControl, realFilename, defaultFileFormat, typeid(ByteT), verbose))
      {
        cerr << "DataServerBodyC::OpenFileReadByte failed to open stream '" << name << "'" << endl;
        return false;
      }

      // Setup raw input sport.
//      DPISPortAttachC<ByteT> iSPortAttachByte(iPortBase, seekControl);
      iSPortByte = DPISPortAttachC<ByteT>(iPortBase, seekControl);

      // Setup inline cache?
      if (cacheSize > 0)
      {
        ONDEBUG(cerr << "DataServerBodyC::OpenFileReadByte added cache size=" << cacheSize << endl);
        iSPortByte = CacheIStreamC<ByteT>(iSPortByte, cacheSize);
      }

      // Setup port share.
      iSPortShareByte = DPISPortShareC<ByteT>(iSPortByte);

      // Set trigger to let us know when people stop using this file.
      iSPortShareByte.TriggerCountZero() = TriggerR(*this, &DataServerVFSRealFileBodyC::ZeroIPortClientsByte);
    }
    else
    {
      iSPortByte = iSPortShareByte.Port();
      iPortBase = DPIPortBaseC(iSPortByte);
    }
    RavlAssert(iSPortShareByte.IsValid());
    RavlAssert(iPortBase.IsValid());
    RavlAssert(iSPortByte.IsValid());

    if (!AddTypeConversion(dataType, iPortBase))
      return false;

    netPort = NetISPortServerBaseC(AttributeCtrlC(iPortBase),
                                   iPortBase,
                                   iSPortByte,
                                   name);

    return true;
  }

  //: Open file and setup cache.
  
  bool DataServerVFSRealFileBodyC::OpenFileWrite(const StringC &typePref) {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenFileWrite, Called. \n");
    MutexLockC lock(access);
    // If file already open ?
    if(oport.IsValid())
      return true;

    // Sort out default type.
    if(!oTypeInfo.IsValid()) {
      StringC useType;
      if(!defaultDataType.IsEmpty())
        useType = defaultDataType;
      else
        useType = typePref;
      oTypeInfo = TypeInfo(RTypeInfo(useType));
      if(!oTypeInfo.IsValid()) {
        cerr << "DataServerVFSRealFileBodyC::OpenFileWrite, type '" << useType << "' unknown. \n";
        return false; 
      }
    }
    
    DPOPortBaseC op;
    DPSeekCtrlC sc;
    if(!OpenOSequenceBase(op,sc,realFilename,defaultFileFormat,oTypeInfo.TypeInfo(),verbose)) {
      cerr << "DataServerVFSRealFileBodyC::OpenFileWrite, Failed to open stream '" << realFilename << "' of type '" << typePref << "' \n";
      return false;
    }
    RavlAssert(op.IsValid());
    DPOPortC<RCWrapAbstractC> aop = DPOSPortAttachC<RCWrapAbstractC>(oTypeInfo.CreateConvToAbstract(op),sc);
    
    if(multiWrite) // Do we need to seralise writes ?
      aop = DPOSerialisePortC<RCWrapAbstractC>(aop);
    
    oport = aop;
    return true;
  }



  bool DataServerVFSRealFileBodyC::AddTypeConversion(const StringC &dataType, DPIPortBaseC& iPort)
  {
    RavlAssert(iTypeInfo.IsValid());
    RavlAssert(iPort.IsValid());

    if (dataType != TypeName(iTypeInfo.TypeInfo()))
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::AddTypeConversion building type converter." << endl);
      DListC<DPConverterBaseC> converterList = SystemTypeConverter().FindConversion(iPort.InputType(), RTypeInfo(dataType));
      if (converterList.IsEmpty())
      {
        cerr << "DataServerVFSRealFileBodyC::AddTypeConversion failed to find type conversion." << endl;
        return false;
      }

      iPort = FileFormatDescC(converterList, true).BuildInputConv(iPort);
    }

    return true;
  }

  //: Close file and discard cache.
  
  bool DataServerVFSRealFileBodyC::CloseIFileAbstract() {
    iSPortShareAbstract.Invalidate();
    return true;
  }



  bool DataServerVFSRealFileBodyC::CloseIFileByte() {
    iSPortShareByte.Invalidate();
    return true;
  }

  //: Close output file 
  
  bool DataServerVFSRealFileBodyC::CloseOFile() {
    oport.Invalidate();
    return true;
  }
  
  //: Called if when output file client disconnect it.
  
  bool DataServerVFSRealFileBodyC::DisconnectOPortClient() {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClient(), Called. Ref=" << oport.References() << "\n");
    MutexLockC lock(access);
    if(oport.IsValid() && oport.References() <= 2) { // Are all client references gone? Last two are member variable and disconnect signal.
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClient(), Dropping output port. \n");
      CloseOFile();
    }
    return true;
  }

  //: Called if when file stop's being used.
  
  bool DataServerVFSRealFileBodyC::ZeroIPortClientsAbstract() {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::ZeroIPortClients, Called \n");
    MutexLockC lock(access);
    CloseIFileAbstract();
    return true;
  }



  bool DataServerVFSRealFileBodyC::ZeroIPortClientsByte() {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::ZeroIPortClients, Called \n");
    MutexLockC lock(access);
    CloseIFileByte();
    return true;
  }



  bool DataServerVFSRealFileBodyC::DeleteOnClose()
  {
    if (deleteOnClose)
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DeleteOnClose" << \
              " oport=" << (oport.IsValid() ? oport.References() : 0) << \
              " iSPortShareAbstract=" << (iSPortShareAbstract.IsValid() ? iSPortShareAbstract.References() : 0) << \
              " iSPortShareByte=" << (iSPortShareByte.IsValid() ? iSPortShareByte.References() : 0) << endl);
      
      const int references = (oport.IsValid() ? oport.References() : 0) + \
                             (iSPortShareAbstract.IsValid() ? iSPortShareAbstract.References() : 0) + \
                             (iSPortShareByte.IsValid() ? iSPortShareByte.References() : 0);
      if (references == 0)
      {
        ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DeleteOnClose deleting '" << realFilename << "'" << endl);
        bool deleted = realFilename.Exists() && realFilename.Remove();

        if (deleted and sigOnDelete.IsValid())
          sigOnDelete(AbsoluteName());
        
        return deleted;
      }
    }

    return false;
  }
  
}
