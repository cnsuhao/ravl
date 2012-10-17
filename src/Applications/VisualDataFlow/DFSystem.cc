// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlVDF
//! author="Charles Galambos"
//! file="Ravl/Applications/VisualDataFlow/DFSystem.cc"

#include "Ravl/DF/DFSystem.hh"
#include "Ravl/DF/DFData.hh"
#include "Ravl/DF/DFPort.hh"
#include "Ravl/DF/DFLink.hh"
#include "Ravl/IO.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/Stream.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlDFN {
  
  //: Constructor.
  
  DFSystemBodyC::DFSystemBodyC(const StringC &nname)
    : name(nname),
      sigChange(DFOU_CHANGED),
      factory(PROJECT_OUT "/share/RAVL/vdf/factory.cfg")
  {}
  
  //: Read from std::istream.
  
  DFSystemBodyC::DFSystemBodyC(std::istream &is)
  { is >> name >> objects; }
  
  //: Read from std::istream.
  
  DFSystemBodyC::DFSystemBodyC(BinIStreamC &is)
  { is >> name >> objects; }
  
  //: Read from XMLStream.
  
  DFSystemBodyC::DFSystemBodyC(XMLIStreamC &is) {
    
  }
  
  //: Save std::ostream.
  
  bool DFSystemBodyC::Save(std::ostream &strm) const {
    strm << name << ' ' << objects << ' ' ;
    return true;
  }
  
  //: Save std::ostream.
  
  bool DFSystemBodyC::Save(BinOStreamC &strm) const {
    strm << name << objects;
    return true;
  }
  
  //: Save std::ostream.
  
  bool DFSystemBodyC::Save(XMLOStreamC &strm) const {
    strm << XMLStartTag("DFSystem") << XMLAttribute("name",name);
    // Save objects first.
    for(DLIterC<DFObjectC> it(objects);it;it++) {
      DFLinkC lnk(*it);
      if(!lnk.IsValid())
	it->Save(strm,true);
    }
    // Save links between them
    for(DLIterC<DFObjectC> it(objects);it;it++) {
      DFLinkC lnk(*it);
      if(lnk.IsValid())
	it->Save(strm,true);
    } 
    strm << XMLEndTag;
    return true;
  }
  
  //: Add an object to the system.
  
  bool DFSystemBodyC::AddObject(DFObjectC &obj) {
    objects += obj;
    sigChange(DFOU_ADDED,obj);
    return true;
  }

  //: Remove an object from the system.
  
  bool DFSystemBodyC::DelObject(DFObjectC &obj) {
    if(objects.Del(obj)) {
      sigChange(DFOU_DELETED,obj);
      return true;
    }
    return false;
  }
  
  //: Create an input port.
  
  DFObjectC DFSystemBodyC::CreateIPort(const StringC &name) {
    DFIPortC ip(name);
    AddObject(ip);
    return ip;
  }
  
  //: Create an output port.
  
  DFObjectC DFSystemBodyC::CreateOPort(const StringC &name) {
    DFIPortC op(name);
    AddObject(op);
    return op;
  }
  
  //: Load object from file and add it the system.
  
  DFObjectC DFSystemBodyC::LoadObject(const StringC &fn) {
    DFObjectC ret;
    ONDEBUG(std::cerr << "DFSystemBodyC::LoadObject(" << fn << ") Called. \n");
    if(!Load(fn,ret))
      return DFObjectC();
    AddObject(ret);
    return ret;
  }
  
  
  //////////////////////////////////////////////////////////////////////////
  
  //: Write to an std::ostream.
  
  std::ostream &operator<<(std::ostream &strm,const DFSystemC &dfa) {
    RavlAssert(dfa.IsValid());
    dfa.Save(strm);
    return strm;
  }
  
  //: Read from an std::istream.
  
  std::istream &operator>>(std::istream &strm,DFSystemC &dfa) {
    dfa = DFSystemC(strm);
    return strm;
  }
  
  //: Write to an std::ostream.
  
  BinOStreamC &operator<<(BinOStreamC &strm,const DFSystemC &dfa) {
    RavlAssert(dfa.IsValid());
    dfa.Save(strm);
    return strm;
  }
  
  //: Read from an std::istream.
  
  BinIStreamC &operator>>(BinIStreamC &strm,DFSystemC &dfa) {
    dfa = DFSystemC(strm);
    return strm;
  }

}
