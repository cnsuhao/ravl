// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#include "Ravl/DataServer/DataServer.hh"
#include "Ravl/CallMethods.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/StringList.hh"
#include "Ravl/BlkQueue.hh"
#include "Ravl/Tuple2.hh"

#include "Ravl/DataServer/DataServerVFSNode.hh"
#include "Ravl/DataServer/DataServerVFSRealFile.hh"
#include "Ravl/DataServer/DataServerVFSRealDir.hh"

#include "Ravl/XMLFactoryRegister.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //: XML Factory constructor.

  DataServerBodyC::DataServerBodyC(const XMLFactoryContextC &factory)
   : NetPortManagerBodyC(factory),
     m_vfs(true),
     m_signalNodeClosed(StringC()),
     m_signalNodeRemoved(StringC())
  {
    SetUnregisterOnDisconnect(true);

    if(!factory.UseChildComponent("Root",m_vfs.Data(),false,typeid(DataServerVFSNodeC))) {
      // Setup root VFS node.
      m_vfs.Data() = DataServerVFSNodeC("","",false,true);
    }

    // Set up delete signal
    m_connectionSet += ConnectPtr(m_signalNodeClosed, CBRefT(this), &DataServerBodyC::OnClose);
    m_connectionSet += ConnectPtr(m_signalNodeRemoved,CBRefT(this), &DataServerBodyC::OnDelete);
  }

  //: Constructor.
  
  DataServerBodyC::DataServerBodyC(const StringC &name)
  : NetPortManagerBodyC(name,true),
    m_vfs(true),
    m_signalNodeClosed(StringC()),
    m_signalNodeRemoved(StringC())
  {
    // Setup root VFS node.
    m_vfs.Data() = DataServerVFSNodeC("","",false,true);

    // Set up delete signal
    m_connectionSet += ConnectPtr(m_signalNodeClosed, CBRefT(this), &DataServerBodyC::OnClose);
    m_connectionSet += ConnectPtr(m_signalNodeRemoved,CBRefT(this), &DataServerBodyC::OnDelete);
  }
  
  //: Open server connection.
  
  bool DataServerBodyC::Open(const StringC &addr) {
    StringC empty;
    if (IsOpen())
      return true;
    RavlDebug("DataServer opening '%s'",addr.data());
    
    RegisterIPortRequestManager(TriggerR(*this,&DataServerBodyC::HandleRequestIPort,empty,empty,NetISPortServerBaseC()));
    RegisterOPortRequestManager(TriggerR(*this,&DataServerBodyC::HandleRequestOPort,empty,empty,NetOSPortServerBaseC()));
    
    return NetPortManagerBodyC::Open(addr);
  }
  
  //: Read a new config file.
  // Build Virtual File System appropriately.
  
  bool DataServerBodyC::ReadConfigFile(const StringC &filename, bool useAddress) {
    ConfigFileC cfg(filename);

    MutexLockC lock(m_access);

    //: Configure the root node.
      
    BlkQueueC<Tuple2C<ConfigFileC,HashTreeC<StringC,DataServerVFSNodeC> > > toDo;
    ConfigFileC rootSection = cfg.Section("VFS");
    RavlAssert(m_vfs.IsValid());
    if(rootSection.IsValid())
      toDo.InsLast(Tuple2C<ConfigFileC,HashTreeC<StringC,DataServerVFSNodeC> >(rootSection,m_vfs));
    
    while(!toDo.IsEmpty()) {
      // Get next thing.
      Tuple2C<ConfigFileC,HashTreeC<StringC,DataServerVFSNodeC> > at = toDo.GetFirst();
      
      // Configure it.
      RavlAssert(at.Data2().Data().IsValid());
      at.Data2().Data().Configure(at.Data1());
      
      // Any children ?
      
      DListC<StringC> children = at.Data1().ListSections();
      if(!at.Data2().Data().IsDirectory() && !children.IsEmpty()) {
        cerr << "DataServerBodyC::ReadConfigFile, WARNING: SubSection's in non directory object '" << at.Data1().Name() << "' \n";
        continue;
      }
      
      for(DLIterC<StringC> it(children);it;it++) {
        ConfigFileC subSection = at.Data1().Section(*it);
        
        // What kind of node is wanted ?
        StringC nodeType = subSection["NodeType"];
        StringC nodePath = at.Data2().Data().AbsoluteName();
        DataServerVFSNodeC node;
        if(nodeType == "" || nodeType == "VDir") { // Virtual directory ?
          node = DataServerVFSNodeC(subSection.Name(),nodePath,false,true);
        } else if(nodeType == "RealFile") {        // Real file ?
          node = DataServerVFSRealFileC(*it,nodePath,subSection.Name());
        } else if(nodeType == "RealDir") {
          node = DataServerVFSRealDirC(*it,nodePath,subSection.Name());
        } else { // Something strange.
          cerr << "DataServerBodyC::ReadConfigFile, WARNING: Unknown node type '" << nodeType << "' in file '" << filename << "'\n";
          continue;
        }
        
        RavlAssert(node.IsValid());
        node.SetCloseSignal(m_signalNodeClosed);
        node.SetDeleteSignal(m_signalNodeRemoved);
        
        // Add new node into tree.
        if(!at.Data2().Add(*it,node)) {
          cerr << "DataServerBodyC::ReadConfigFile, WARNING: Duplicate node entry '" << *it << "' in file '" << filename << "'\n";
          continue;
        }
        
        // Retrieve new node entry.
        HashTreeC<StringC,DataServerVFSNodeC>  entry;
        at.Data2().Child(*it,entry);
        RavlAssert(entry.IsValid());
        
        // Put it on the todo list.
        toDo.InsLast(Tuple2C<ConfigFileC,HashTreeC<StringC,DataServerVFSNodeC> >(subSection,entry));
      }
    }

    if (useAddress)
    {
      StringC hostAddr = cfg["Host"];
      if(hostAddr.IsEmpty()) {
        cerr << "DataServerBodyC::ReadConfigFile, Not host address given. ";
        return false;
      }

      return Open(hostAddr);
    }

    return true;
  }
  
  //: Find a virtual file system node.
  // Returns true if node found successfully.
  
  bool DataServerBodyC::FindVFSNode(const StringC &vfilename,HashTreeNodeC<StringC,DataServerVFSNodeC> &vfs,DListC<StringC> &remainingPath) {
    StringListC path(vfilename,"/");
    HashTreeNodeC<StringC,DataServerVFSNodeC> at = m_vfs;
    DLIterC<StringC> it(path);
    for(;it && !at.IsLeaf();it++) {
      // Is this a node in the tree with children ?
      HashTreeC<StringC,DataServerVFSNodeC> ht(at);
      RavlAssert(ht.IsValid()); // No children.
      if(!ht.Child(*it,at))
        break; // Child not found.
    }
    
    // There can only be a remainder in the path if the last node in the tree is a directory.
    if(!at.Data().IsDirectory() && it)
      return false;
    
    // Retrieve node of interest.
    vfs = at;
    
    // Sort out any unused path items.
    if(it)
      remainingPath = it.InclusiveTail();
    else
      remainingPath.Empty();
    ONDEBUG(cerr << "DataServerBodyC::FindVFSNode, '" << vfilename << "' -> '" << vfs.Data().Name() << "' PathSize=" << remainingPath.Size() << "\n");
    return true;
  }



  void DataServerBodyC::ZeroOwners()
  {
    m_connectionSet.DisconnectAll();
    m_vfs.Invalidate();

    NetPortManagerBodyC::ZeroOwners();
  }



  bool DataServerBodyC::AddNode(const StringC& path, const StringC& nodeType, const HashC<StringC, StringC>& options)
  {
    MutexLockC lock(m_access);

    HashTreeNodeC<StringC, DataServerVFSNodeC> foundNode;
    DListC<StringC> remainingPath;
    if (FindVFSNode(path, foundNode, remainingPath))
    {
      RavlAssert(foundNode.IsValid());

      if (foundNode.Data().DeletePending())
      {
        cerr << "DataServerBodyC::AddNode cannot add to node marked for deletion" << endl;
        return false;
      }

      if (remainingPath.Size() == 0)
      {
        cerr << "DataServerBodyC::AddNode path exists" << endl;
        return false;
      }

      if (remainingPath.Size() > 1)
      {
        cerr << "DataServerBodyC::AddNode not all leading path elements exist" << endl;
        return false;
      }

      StringC virtualName = remainingPath.PopFirst();
      StringC virtualPath = foundNode.Data().AbsoluteName();
      ONDEBUG(cerr << "DataServerBodyC::AddNode path='" << path << "' virtualName='" << virtualName.chars() << "' virtualPath='" << virtualPath << "'" << endl);
      DataServerVFSNodeC newNode;
      if (nodeType == "" || nodeType == "VDir")
      {
        // Virtual directory
        newNode = DataServerVFSNodeC(virtualName, virtualPath, false, true);
      }
      else
      {
        // Writeable?
        StringC canWriteStr;
        options.Lookup("CanWrite", canWriteStr);
        bool canWrite = (canWriteStr == "1");

        StringC realName;
        if (options.Lookup("Filename", realName))
        {
          ONDEBUG(cerr << "DataServerBodyC::AddNode realName '" << realName.chars() << "'" << endl);
          if(nodeType == "RealFile")
          {
            // File
            DataServerVFSRealFileC newFileNode(virtualName, virtualPath, realName, canWrite);

            StringC fileFormatStr;
            if (options.Lookup("FileFormat", fileFormatStr))
              newFileNode.SetFileFormat(fileFormatStr);

            newNode = newFileNode;
          }
          else
          {
            if (nodeType == "RealDir")
            {
              // Directory
              StringC canCreateStr;
              options.Lookup("CanCreate", canCreateStr);
              bool canCreate = (canCreateStr == "1");

              DataServerVFSRealDirC newDirectoryNode(virtualName, virtualPath, realName, canWrite, canCreate);

              StringC fileFormatStr;
              if (options.Lookup("FileFormat", fileFormatStr))
                newDirectoryNode.SetFileFormat(fileFormatStr);

              newNode = newDirectoryNode;
            }
            else
            {
              cerr << "DataServerBodyC::AddNode unknown type '" << nodeType << "' for node '" << path << endl;
              return false;
            }
          }
        }
        else
        {
          cerr << "DataServerBodyC::AddNode real nodes '" << path << "' need a filename" << endl;
          return false;
        }
      }

      RavlAssert(newNode.IsValid());
      newNode.SetCloseSignal(m_signalNodeClosed);
      newNode.SetDeleteSignal(m_signalNodeRemoved);
      
      HashTreeC<StringC, DataServerVFSNodeC> vfsTree(foundNode);
      return vfsTree.Add(virtualName, newNode);
    }

    return false;
  }

  bool DataServerBodyC::RemoveNode(const StringC& path, bool removeFromDisk)
  {
    MutexLockC lock(m_access);

    HashTreeNodeC<StringC, DataServerVFSNodeC> foundNode = m_vfs;
    StringListC pathList(path, "/");
    DLIterC<StringC> pathListIter(pathList);
    for (; pathListIter; pathListIter++)
    {
      HashTreeC<StringC, DataServerVFSNodeC> vfsTree(foundNode);
      RavlAssert(vfsTree.IsValid());
      if (vfsTree.Child(*pathListIter, foundNode))
      {
        RavlAssert(foundNode.IsValid());

        if (!pathListIter.IsLast())
        {
          if (foundNode.Data().RemovePending() || foundNode.Data().DeletePending())
          {
            ONDEBUG(cerr << "DataServerBodyC::RemoveNode parent node of (" << path << ") marked for removal or deletion" << endl);

            return true;
          }

          continue;
        }

        if (!(foundNode.Data().IsDirectory() && foundNode.Data().PortsOpen()))
        {
          ONDEBUG(cerr << "DataServerBodyC::RemoveNode removing node from tree (" << path << ")" << endl);

          if (!vfsTree.Remove(*pathListIter))
          {
            cerr << "DataServerBodyC::RemoveNode failed to remove node from tree '" << path << "'" << endl;
            break;
          }
        }
        else
        {
          ONDEBUG(cerr << "DataServerBodyC::RemoveNode marking node for removal (" << path << ")" << endl);

          foundNode.Data().SetRemovePending(true);
        }

        lock.Unlock();

        if (removeFromDisk)
        {
          return foundNode.Data().Delete();
        }

        return true;
      }
      else
      {
        RavlAssert(foundNode.IsValid());

        if (foundNode.Data().IsDirectory())
        {
          if (foundNode.Data().RemovePending() || foundNode.Data().DeletePending())
          {
            ONDEBUG(cerr << "DataServerBodyC::RemoveNode parent node of (" << path << ") marked for removal or deletion" << endl);

            return true;
          }

          lock.Unlock();
          
          if (!removeFromDisk)
          {
            cerr << "DataServerBodyC::RemoveNode remove from directory failed as file deletion not enabled for '" << path << "'" << endl;
            break;
          }

          DListC<StringC> remainingPath = pathListIter.InclusiveTail();
          return foundNode.Data().Delete(remainingPath);
        }

        cerr << "DataServerBodyC::RemoveNode failed to find node '" << path << "'" << endl;
        break;
      }
    }

    return false;
  }

  bool DataServerBodyC::QueryNodeSpace(const StringC& path, Int64T& total, Int64T& used, Int64T& available)
  {
    MutexLockC lock(m_access);

    HashTreeNodeC<StringC, DataServerVFSNodeC> foundNode = m_vfs;
    StringListC pathList(path, "/");
    DLIterC<StringC> pathListIter(pathList);
    for (; pathListIter; pathListIter++)
    {
      HashTreeC<StringC, DataServerVFSNodeC> vfsTree(foundNode);
      RavlAssert(vfsTree.IsValid());
      if (vfsTree.Child(*pathListIter, foundNode))
      {
        if (!pathListIter.IsLast())
        {
          continue;
        }

        StringListC remainingPath = pathListIter.Tail();
        return foundNode.Data().QueryNodeSpace(remainingPath.Cat("/"), total, used, available);
      }
      else
      {
        if (foundNode.IsValid() && foundNode.Data().IsDirectory())
        {
          lock.Unlock();

          StringListC remainingPath = pathListIter.InclusiveTail();
          return foundNode.Data().QueryNodeSpace(remainingPath.Cat("/"), total, used, available);
        }

        cerr << "DataServerBodyC::QueryNodeSpace failed to find node '" << path << "'" << endl;
        break;
      }
    }

    return false;
  }

  //: Handle a request for an input port.
  
  bool DataServerBodyC::HandleRequestIPort(StringC name,StringC dataType,NetISPortServerBaseC &port) {
    ONDEBUG(cerr << "DataServerBodyC::HandleRequestIPort, Name=" << name << " Type=" << dataType << "\n");
    
    MutexLockC lock(m_access);

    HashTreeNodeC<StringC,DataServerVFSNodeC> foundNode;
    DListC<StringC> remainingPath;
    if(!FindVFSNode(name,foundNode,remainingPath)) {
      cerr << "DataServerBodyC::HandleRequestIPort, Failed to find VFSNode for '" << name << "'\n";
      return false;
    }

    lock.Unlock();

    RavlAssert(foundNode.IsValid());
    if(!foundNode.Data().OpenIPort(remainingPath,dataType,port)) {
      cerr << "DataServerBodyC::HandleRequestIPort, Failed to open file '" << name << "'\n";
      return false;
    }
    return true;
  }
  
  //: Handle a request for an output port.
  
  bool DataServerBodyC::HandleRequestOPort(StringC name,StringC dataType,NetOSPortServerBaseC &port) {
    ONDEBUG(cerr << "DataServerBodyC::HandleRequestOPort, Name=" << name << " Type=" << dataType << "\n");
    
    MutexLockC lock(m_access);

    HashTreeNodeC<StringC,DataServerVFSNodeC> foundNode;
    DListC<StringC> remainingPath;
    if(!FindVFSNode(name,foundNode,remainingPath)) {
      cerr << "DataServerBodyC::HandleRequestIPort, Failed to find VFSNode for '" << name << "'\n";
      return false;
    }

    lock.Unlock();

    RavlAssert(foundNode.IsValid());
    return foundNode.Data().OpenOPort(remainingPath,dataType,port);
  }
  


  bool DataServerBodyC::OnClose(StringC& path)
  {
    return OnCloseOrDelete(path, false);
  }



  bool DataServerBodyC::OnDelete(StringC& path)
  {
    return OnCloseOrDelete(path, true);
  }



  bool DataServerBodyC::OnCloseOrDelete(StringC &path, bool onDelete)
  {
    ONDEBUG(cerr << "DataServerBodyC::OnCloseOrDelete path (" << path << ") action (" << (onDelete ? "delete" : "close") << ")" << endl);

    MutexLockC lock(m_access);

    HashTreeNodeC<StringC, DataServerVFSNodeC> foundNode;
    DListC<StringC> remainingPath;
    if (FindVFSNode(path, foundNode, remainingPath))
    {
      ONDEBUG(cerr << "DataServerBodyC::OnCloseOrDelete found node for path (" << path << ")" << endl);

      lock.Unlock();

      RavlAssert(foundNode.IsValid());
      bool signalValue = false;
      if (onDelete)
        signalValue = foundNode.Data().OnDelete(remainingPath);
      else
        signalValue = foundNode.Data().OnClose(remainingPath);

      PruneRemovedNodes(path);

      return signalValue;
    }

    return false;
  }

  bool DataServerBodyC::PruneRemovedNodes(StringC &path)
  {
    ONDEBUG(cerr << "DataServerBodyC::PruneRemovedNodes path (" << path << ")" << endl);

    MutexLockC lock(m_access);

    HashTreeNodeC<StringC, DataServerVFSNodeC> foundNode = m_vfs;
    StringListC pathList(path, "/");
    DLIterC<StringC> pathListIter(pathList);

    for (; pathListIter; pathListIter++)
    {
      HashTreeC<StringC, DataServerVFSNodeC> vfsTree(foundNode);
      RavlAssert(vfsTree.IsValid());
      if (vfsTree.Child(*pathListIter, foundNode))
      {
        RavlAssert(foundNode.IsValid());

        if (foundNode.Data().IsDirectory() && foundNode.Data().RemovePending() && !foundNode.Data().PortsOpen())
        {
          ONDEBUG(cerr << "DataServerBodyC::PruneRemovedNodes removing node from tree (" << path << ")" << endl);
          if (!vfsTree.Remove(*pathListIter))
          {
            cerr << "DataServerBodyC::PruneRemovedNodes failed to remove node from tree '" << path << "'" << endl;
            break;
          }

          return true;
        }

        if (!pathListIter.IsLast())
        {
          continue;
        }
      }

      break;
    }

    return false;
  }

  void LinkDataServer()
  {}

  XMLFactoryRegisterHandleConvertC<DataServerC,NetPortManagerC> g_registerXMLFactoryDataServer("RavlN::DataServerC");

}
