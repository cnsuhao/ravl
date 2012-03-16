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
#include "Ravl/Calls.hh"
#include "Ravl/XMLFactoryRegister.hh"


#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  
  //: XMLFactory Constructor

  DataServerVFSRealFileBodyC::DataServerVFSRealFileBodyC(const XMLFactoryContextC &factory)
   : DataServerVFSNodeBodyC(factory),
     defaultDataType(factory.AttributeString("defaultDataType","")),
     defaultFileFormat(factory.AttributeString("defaultFileFormat","")),
     cacheSize(factory.AttributeInt64("cacheSize",0)),
     realFilename(factory.AttributeString("realFilename","")),
     canSeek(factory.AttributeBool("canSeek",true)),
     multiWrite(factory.AttributeBool("multiWrite",false))
  {

  }

  //: Constructor.
  
  DataServerVFSRealFileBodyC::DataServerVFSRealFileBodyC(const StringC &vname,const StringC& npath,const StringC &nRealFilename,bool canWrite)
    : DataServerVFSNodeBodyC(vname,npath,canWrite,false),
      cacheSize(0),
      realFilename(nRealFilename),
      canSeek(true),
      multiWrite(false)
  {
    ONDEBUG(std::cerr << "DataServerVFSRealFileBodyC::DataServerVFSRealFileBodyC (" << this << ") Called name=" << name << " path=" << path << "\n");
  }
  
  //: Destructor.
  
  DataServerVFSRealFileBodyC::~DataServerVFSRealFileBodyC()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::~DataServerVFSRealFileBodyC (" << this << ")\n");

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::~DataServerVFSRealFileBodyC " << \
                    "iSPortShareAbstract (" << (iSPortShareAbstract.IsValid() ? static_cast<Int64T>(iSPortShareAbstract.References()) : -1) << ") " << \
                    "iSPortShareByte (" << (iSPortShareByte.IsValid() ? static_cast<Int64T>(iSPortShareByte.References()) : -1) << ") " << \
                    "oPortAbstract (" << (oPortAbstract.IsValid() ? static_cast<Int64T>(oPortAbstract.References()) : -1) << ") " << \
                    "oPortByte (" << (oPortByte.IsValid() ? static_cast<Int64T>(oPortByte.References()) : -1) << ") " << endl);

    CloseIFileAbstract();
    CloseIFileByte();
    CloseOFileAbstract();
    CloseOFileByte();
    
    if (deletePending)
    	DoDelete();
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

    MutexLockC lock(access);

    if (oPortAbstract.IsValid() || oPortByte.IsValid())
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
    RavlAssert(iTypeInfo.IsValid());

    // Choose the correct input port
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenIPort dataType='" << dataType << "' iTypeInfo='" << iTypeInfo.Name() << "' defaultFileFormat='" << defaultFileFormat << "'" << endl);
    if (iTypeInfo.TypeInfo() == typeid(ByteT) && defaultFileFormat == "bytefile")
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenIPort opening byte file for '" << name << "'" << endl);
      return OpenFileReadByte(dataType, port);
    }

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenIPort opening abstract file for '" << name << "'" << endl);
    return OpenFileReadAbstract(dataType, port);
  }
  
  //: Open output port.
  
  bool DataServerVFSRealFileBodyC::OpenOPort(DListC<StringC> &remainingPath, const StringC &dataType, NetOSPortServerBaseC &port)
  {
    if (!remainingPath.IsEmpty())
    {
      cerr << "DataServerVFSRealFileBodyC::OpenOPort failed to open as path remaining after a valid filename." << endl;
      return false;
    }
    
    MutexLockC lock(access);

    if (iSPortShareAbstract.IsValid() || iSPortShareByte.IsValid())
    {
      cerr << "DataServerVFSRealFileBodyC::OpenOPort failed to open for writing, as already open for reading." << endl;
      return false;
    }

    // Sort out default type.
    if (!oTypeInfo.IsValid())
    {
      StringC useType;
      if (!defaultDataType.IsEmpty())
        useType = defaultDataType;
      else
        useType = dataType;

      oTypeInfo = TypeInfo(RTypeInfo(useType));
      if (!oTypeInfo.IsValid())
      {
        cerr << "DataServerVFSRealFileBodyC::OpenOPort failed to open as type '" << useType << "' unknown" << endl;
        return false;
      }
    }
    RavlAssert(oTypeInfo.IsValid());

    // Choose the correct output port
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenOPort dataType='" << dataType << "' oTypeInfo='" << oTypeInfo.Name() << "' defaultFileFormat='" << defaultFileFormat << "'" << endl);
    if (oTypeInfo.TypeInfo() == typeid(ByteT) && defaultFileFormat == "bytefile")
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenOPort opening byte file for '" << name << "'" << endl);
      return OpenFileWriteByte(dataType, port);
    }

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::OpenOPort opening abstract file for '" << name << "'" << endl);
    return OpenFileWriteAbstract(dataType, port);
  }



  bool DataServerVFSRealFileBodyC::Delete()
  {
    if (!CanWrite())
    {
      cerr << "DataServerVFSRealFileBodyC::Delete trying to delete read-only file '" << name << "'" << endl;
      return false;
    }

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::Delete marking for delete" << endl);
    deletePending = true;
    return true;
  }



  bool DataServerVFSRealFileBodyC::QueryNodeSpace(const StringC& remainingPath, Int64T& total, Int64T& used, Int64T& available)
  {
    if (!remainingPath.IsEmpty())
    {
      cerr << "DataServerVFSRealFileBodyC::QueryNodeSpace failed to open as path remaining after a valid filename." << endl;
      return false;
    }

    MutexLockC lock(access);

    if (!realFilename.Exists() || realFilename.IsDirectory() || !realFilename.IsRegular() || !realFilename.IsReadable())
    {
      cerr << "DataServerVFSRealFileBodyC::QueryNodeSpace cannot query file size for '" << realFilename << "'." << endl;
      return false;
    }

    total = -1;
    available = -1;
    used = realFilename.FileSize();
    
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

      // Convert into an abstract stream.
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
      DataServerVFSRealFileC ref(*this);
      iSPortShareAbstract.TriggerCountZero() = Trigger(ref, &DataServerVFSRealFileC::ZeroIPortClientsAbstract);
    }
    RavlAssert(iSPortShareAbstract.IsValid());

    DPISPortC<RCWrapAbstractC> iSPortAbstract = iSPortShareAbstract;
    DPIPortBaseC iPortBase = iTypeInfo.CreateConvFromAbstract(iSPortAbstract);

    if (!AddIPortTypeConversion(dataType, iPortBase))
      return false;

    netPort = NetISPortServerBaseC(iPortBase,
                                   iPortBase,
                                   iSPortAbstract,
                                   name);

#if DODEBUG
    DataServerVFSRealFileC ref(*this);
    Connect(netPort.SigConnectionClosed(), ref, &DataServerVFSRealFileC::DisconnectIPortClientAbstract);
#endif

    return true;
  }



  bool DataServerVFSRealFileBodyC::OpenFileReadByte(const StringC& dataType, NetISPortServerBaseC& netPort)
  {
    if (!iSPortShareByte.IsValid())
    {
      DPIPortBaseC iPortBase;
      DPISPortC<ByteT> iSPortByte;
      DPSeekCtrlC seekControl;
      if (!OpenISequenceBase(iPortBase, seekControl, realFilename, defaultFileFormat, typeid(ByteT), verbose))
      {
        std::cerr << "DataServerBodyC::OpenFileReadByte failed to open stream '" << name << "'" << std::endl;
        return false;
      }

      // Setup raw input sport.
      iSPortByte = DPISPortAttachC<ByteT>(iPortBase, seekControl);

      // Setup inline cache?
      if (cacheSize > 0)
      {
        ONDEBUG(std::cerr << "DataServerBodyC::OpenFileReadByte added cache size=" << cacheSize << std::endl);
        iSPortByte = CacheIStreamC<ByteT>(iSPortByte, cacheSize);
      }

      // Setup port share.
      iSPortShareByte = DPISPortShareC<ByteT>(iSPortByte);

      // Set trigger to let us know when people stop using this file.
      DataServerVFSRealFileC ref(*this);
      iSPortShareByte.TriggerCountZero() = Trigger(ref, &DataServerVFSRealFileC::ZeroIPortClientsByte);
    }
    RavlAssert(iSPortShareByte.IsValid());

    DPISPortC<ByteT> iSPortByte = iSPortShareByte;

    if (!AddIPortTypeConversion(dataType, iSPortByte))
      return false;

    netPort = NetISPortServerBaseC(iSPortByte,
                                   iSPortByte,
                                   iSPortByte,
                                   name);

#if DODEBUG
    DataServerVFSRealFileC ref(*this);
    Connect(netPort.SigConnectionClosed(), ref, &DataServerVFSRealFileC::DisconnectIPortClientByte);
#endif

    return true;
  }

  //: Open file and setup cache.
  
  bool DataServerVFSRealFileBodyC::OpenFileWriteAbstract(const StringC &dataType, NetOSPortServerBaseC& netPort)
  {
    if (oPortByte.IsValid())
    {
      cerr << "DataServerBodyC::OpenFileWriteAbstract failed to open abstract stream '" << name << "' as byte stream already open" << endl;
      return false;
    }

    if (!oPortAbstract.IsValid())
    {
      DPOPortBaseC oPortBase;
      DPSeekCtrlC seekControl;
      if (!OpenOSequenceBase(oPortBase, seekControl, realFilename, defaultFileFormat, oTypeInfo.TypeInfo(), verbose))
      {
        cerr << "DataServerBodyC::OpenFileWriteAbstract failed to open stream '" << name << "' of type '" << oTypeInfo.Name() << "'" << endl;
        return false;
      }

      // Convert into an abstract stream.
      oPortAbstract = oTypeInfo.CreateConvToAbstract(oPortBase);

      // Setup raw input sport.
      oPortAbstract = DPOSPortAttachC<RCWrapAbstractC>(oPortAbstract, seekControl);

      if (multiWrite)
      {
        ONDEBUG(cerr << "DataServerBodyC::OpenFileWriteAbstract adding port serialisation" << endl);
        oPortAbstract = DPOSerialisePortC<RCWrapAbstractC>(oPortAbstract);
      }
    }
    RavlAssert(oPortAbstract.IsValid());
    
    if (oPortAbstract.References() > 1 && !multiWrite)
    {
      cerr << "DataServerVFSRealFileBodyC::OpenFileWriteAbstract failed as port already open and multi-write not enabled" << endl;
      return false;
    }

    DPOPortBaseC oPortBase = oTypeInfo.CreateConvFromAbstract(oPortAbstract);

    if (!AddOPortTypeConversion(dataType, oPortBase))
      return false;

    netPort = NetOSPortServerBaseC(oPortBase,
                                   oPortBase,
                                   oPortBase,
                                   name);

    DataServerVFSRealFileC ref(*this);
    Connect(netPort.SigConnectionClosed(), ref, &DataServerVFSRealFileC::DisconnectOPortClientAbstract);
    
    return true;
  }



  bool DataServerVFSRealFileBodyC::OpenFileWriteByte(const StringC &dataType, NetOSPortServerBaseC& netPort)
  {
    if (oPortAbstract.IsValid())
    {
      cerr << "DataServerBodyC::OpenFileWriteAbstract failed to open byte stream '" << name << "' as abstract stream already open" << endl;
      return false;
    }

    if (!oPortByte.IsValid())
    {
      DPOPortBaseC oPortBase;
      DPSeekCtrlC seekControl;
      if (!OpenOSequenceBase(oPortBase, seekControl, realFilename, defaultFileFormat, typeid(ByteT), verbose))
      {
        cerr << "DataServerBodyC::OpenFileWriteByte failed to open stream '" << name << "'" << endl;
        return false;
      }

      // Setup raw input sport.
      oPortByte = DPOSPortAttachC<ByteT>(oPortBase, seekControl);

      if (multiWrite)
      {
        ONDEBUG(cerr << "DataServerBodyC::OpenFileWriteByte adding port serialisation" << endl);
        oPortByte = DPOSerialisePortC<ByteT>(oPortByte);
      }
    }
    RavlAssert(oPortByte.IsValid());

    if (oPortByte.References() > 1 && !multiWrite)
    {
      cerr << "DataServerVFSRealFileBodyC::OpenFileWriteByte failed as port already open and multi-write not enabled" << endl;
      return false;
    }

    DPOPortBaseC oPortBase = oPortByte;

    if (!AddOPortTypeConversion(dataType, oPortBase))
      return false;

    netPort = NetOSPortServerBaseC(oPortBase,
                                   oPortBase,
                                   oPortBase,
                                   name);

    DataServerVFSRealFileC ref(*this);
    Connect(netPort.SigConnectionClosed(), ref, &DataServerVFSRealFileC::DisconnectOPortClientByte);

    return true;
  }



  bool DataServerVFSRealFileBodyC::AddIPortTypeConversion(const StringC &dataType, DPIPortBaseC& iPort)
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



  bool DataServerVFSRealFileBodyC::AddOPortTypeConversion(const StringC &dataType, DPOPortBaseC& oPort)
  {
    RavlAssert(oTypeInfo.IsValid());
    RavlAssert(oPort.IsValid());

    if (dataType != TypeName(oTypeInfo.TypeInfo()))
    {
      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::AddOPortTypeConversion building type converter." << endl);
      DListC<DPConverterBaseC> converterList = SystemTypeConverter().FindConversion(RTypeInfo(dataType), oPort.OutputType());
      if (converterList.IsEmpty())
      {
        cerr << "DataServerVFSRealFileBodyC::AddOPortTypeConversion failed to find type conversion." << endl;
        return false;
      }

      oPort = FileFormatDescC(converterList, false).BuildOutputConv(oPort);
    }

    return true;
  }

  //: Close file and discard cache.
  
  bool DataServerVFSRealFileBodyC::CloseIFileAbstract()
  {
    RavlAssert(!iSPortShareByte.IsValid());

    MutexLockC lock(access);

    if (iSPortShareAbstract.IsValid())
    {
      iSPortShareAbstract.Invalidate();

      if (sigOnClose.IsValid())
      {
        lock.Unlock();

        sigOnClose(AbsoluteName());
      }
    }

    return true;
  }



  bool DataServerVFSRealFileBodyC::CloseIFileByte()
  {
    RavlAssert(!iSPortShareAbstract.IsValid());

    MutexLockC lock(access);

    if (iSPortShareByte.IsValid())
    {
      iSPortShareByte.Invalidate();

      if (sigOnClose.IsValid())
      {
        lock.Unlock();

        sigOnClose(AbsoluteName());
      }
    }

    return true;
  }

  //: Close output file 
  
  bool DataServerVFSRealFileBodyC::CloseOFileAbstract()
  {
    MutexLockC lock(access);

    if (oPortAbstract.IsValid())
    {
      oPortAbstract.PutEOS();
      oPortAbstract.Invalidate();

      if (sigOnClose.IsValid())
      {
        lock.Unlock();

        sigOnClose(AbsoluteName());
      }
    }

    return true;
  }



  bool DataServerVFSRealFileBodyC::CloseOFileByte()
  {
    MutexLockC lock(access);

    if (oPortByte.IsValid())
    {
      oPortByte.PutEOS();
      oPortByte.Invalidate();

      if (sigOnClose.IsValid())
      {
        lock.Unlock();

        sigOnClose(AbsoluteName());
      }
    }

    return true;
  }



  bool DataServerVFSRealFileBodyC::DisconnectIPortClientAbstract()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectIPortClientAbstract refs (" << (iSPortShareAbstract.IsValid() ? iSPortShareAbstract.References() : -1) << ")" << endl);

    return true;
  }



  bool DataServerVFSRealFileBodyC::DisconnectIPortClientByte()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectIPortClientByte refs (" << (iSPortShareByte.IsValid() ? iSPortShareByte.References() : -1) << ")" << endl);

    return true;
  }


  
  bool DataServerVFSRealFileBodyC::DisconnectOPortClientAbstract()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClientAbstract(), Called. Ref=" << (oPortAbstract.IsValid() ? oPortAbstract.References() : -1) << "\n");

    MutexLockC lock(access);

    // Are all client references gone? Last two are member variable and disconnect signal.
    if (oPortAbstract.IsValid() && oPortAbstract.References() <= 2)
    {
      lock.Unlock();

      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClientAbstract(), Dropping output port. \n");
      CloseOFileAbstract();
    }

    return true;
  }



  bool DataServerVFSRealFileBodyC::DisconnectOPortClientByte()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClientByte(), Called. Ref=" << (oPortByte.IsValid() ? oPortByte.References() : -1) << "\n");

    MutexLockC lock(access);

    // Are all client references gone? Last two are member variable and disconnect signal.
    if (oPortByte.IsValid() && oPortByte.References() <= 2)
    {
      lock.Unlock();

      ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DisconnectOPortClientByte(), Dropping output port. \n");
      CloseOFileByte();
    }
    
    return true;
  }


  
  bool DataServerVFSRealFileBodyC::ZeroIPortClientsAbstract()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::ZeroIPortClientsAbstract (" << this << ")" << endl);

    CloseIFileAbstract();

    return true;
  }



  bool DataServerVFSRealFileBodyC::ZeroIPortClientsByte()
  {
    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::ZeroIPortClientsByte (" << this << ")" << endl);

    CloseIFileByte();

    return true;
  }



  bool DataServerVFSRealFileBodyC::DoDelete()
  {
    RavlAssert(deletePending);

    ONDEBUG(cerr << "DataServerVFSRealFileBodyC::DoDelete deleting '" << realFilename << "'" << endl);
    bool deleted = realFilename.Exists() && realFilename.Remove();

    if (deleted && sigOnDelete.IsValid())
      sigOnDelete(AbsoluteName());

    return deleted;
  }
  
  XMLFactoryRegisterHandleC<DataServerVFSRealFileC> g_registerXMLFactoryDataServerVFSRealFile("RavlN::DataServerVFSRealFileC");

}
