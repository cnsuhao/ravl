// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlIO
//! file="Ravl/Core/IO/StreamOp.cc"

#include "Ravl/DP/StreamOp.hh"

namespace RavlN {

  DPStreamOpBodyC::DPStreamOpBodyC()
  {}
  //: Default constructor.

  DPStreamOpBodyC::DPStreamOpBodyC(const StringC &entityName)
   : DPEntityBodyC(entityName)
  {}
  //: Constructor.

  DPStreamOpBodyC::DPStreamOpBodyC(std::istream &in)
    : DPEntityBodyC(in)
  {}
  //: Stream constructor.
  
  DPStreamOpBodyC::DPStreamOpBodyC(BinIStreamC &in)
    : DPEntityBodyC(in)
  {}
  //: Binary stream constructor.

  StringC DPStreamOpBodyC::OpName() const
  { return EntityName(); }
  //: Op type name.

  //: Input plugs.
  
  DListC<DPIPlugBaseC> DPStreamOpBodyC::IPlugs() const {
    return DListC<DPIPlugBaseC>();
  }
  
  //: Output plugs
  
  DListC<DPOPlugBaseC> DPStreamOpBodyC::OPlugs() const {
    return DListC<DPOPlugBaseC>();
  }
  
  //: Input ports.
  
  DListC<DPIPortBaseC> DPStreamOpBodyC::IPorts() const {
    return DListC<DPIPortBaseC>();
  }
    
  //: Output ports
  
  DListC<DPOPortBaseC> DPStreamOpBodyC::OPorts() const {
    return DListC<DPOPortBaseC>();
  }

  //: Get input
  bool DPStreamOpBodyC::GetIPlug(const StringC &name,DPIPlugBaseC &port)
  {
    DListC<DPIPlugBaseC> plugs = IPlugs();
    for(DLIterC<DPIPlugBaseC> it(plugs);it;it++) {
      if(it->EntityName() == name) {
        port = *it;
        return true;
      }
    }
    return false;

  }


  //: Get output
  bool DPStreamOpBodyC::GetOPlug(const StringC &name,DPOPlugBaseC &port)
  {
    DListC<DPOPlugBaseC> plugs = OPlugs();
    for(DLIterC<DPOPlugBaseC> it(plugs);it;it++) {
      if(it->EntityName() == name) {
        port = *it;
        return true;
      }
    }
    return false;
  }


  //: Set an input
  bool DPStreamOpBodyC::SetIPort(const StringC &name,const DPIPortBaseC &port)
  {
    DListC<DPIPlugBaseC> plugs = IPlugs();
    for(DLIterC<DPIPlugBaseC> it(plugs);it;it++) {
      if(it->EntityName() == name) {
        return it->ConnectPort(port);
      }
    }
    return false;
  }

  //: Set an output
  bool DPStreamOpBodyC::SetOPort(const StringC &name,const DPOPortBaseC &port)
  {
    DListC<DPOPlugBaseC> plugs = OPlugs();
    for(DLIterC<DPOPlugBaseC> it(plugs);it;it++) {
      if(it->EntityName() == name) {
        return it->ConnectPort(port);
      }
    }
    return false;
  }

  //: Get output
  bool DPStreamOpBodyC::GetOPort(const StringC &name,DPOPortBaseC &port) {
    DListC<DPOPortBaseC> ports= OPorts();
    for(DLIterC<DPOPortBaseC> it(ports);it;it++) {
      if(it->EntityName() == name) {
        port = *it;
        return true;
      }
    }
    return false;
  }

  //: Get input
  bool DPStreamOpBodyC::GetIPort(const StringC &name,DPIPortBaseC &port) {
    DListC<DPIPortBaseC> ports = IPorts();
    for(DLIterC<DPIPortBaseC> it(ports);it;it++) {
      if(it->EntityName() == name) {
        port = *it;
        return true;
      }
    }
    return false;
  }

  
}
