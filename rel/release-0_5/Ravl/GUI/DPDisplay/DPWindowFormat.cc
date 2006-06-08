// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlDPDisplay
//! file="Ravl/GUI/DPDisplay/DPWindowFormat.cc"

#include "Ravl/GUI/DPWindowFormat.hh"
#include "Ravl/GUI/DPWindowOPort.hh"
#include "Ravl/Threads/RWLock.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/Hash.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlGUIN {
  
  void InitDPWindowFormat()
  {}
  
  static HashC<StringC,DPWindowC> windows; // List of windows open.
  static RWLockC windowsLock;

  
  //: Default constructor.
  
  DPWindowFormatBodyC::DPWindowFormatBodyC()
  {}
    
  //: Probe for Save.
  
  const type_info &DPWindowFormatBodyC::ProbeSave(const StringC &filename,const type_info &obj_type,bool forceFormat) const {
    ONDEBUG(cerr << "DPWindowFormatBodyC::ProbeSave(), Called. Filename=" << filename << " obj_type=" << TypeName(obj_type) << " ForceFormat=" << forceFormat << "\n");
    if(forceFormat)
      return typeid(DPDisplayObjC);
    if(filename[0] != '@')
      return typeid(void);
    StringC device = ExtractDevice(filename);
    if(device != "X" && device != "XA")
      return typeid(void);
    return typeid(DPDisplayObjC);
  }
  
  //: Create a output port for saving.
  // Will create an Invalid port if not supported.
  
  DPOPortBaseC DPWindowFormatBodyC::CreateOutput(const StringC &filename,const type_info &obj_type) const {
    ONDEBUG(cerr << "DPWindowFormatBodyC::CreateOutput(), Called. Filename=" << filename << " obj_type=" << TypeName(obj_type) << " \n");
    if(obj_type != typeid(DPDisplayObjC))
      return DPOPortBaseC();
    StringC winName = ExtractParams(filename);
    StringC device = ExtractDevice(filename);
    
    // See if the window alread exists.
    RWLockHoldC hold(windowsLock,false);
    DPWindowC win;
    if(!windows.Lookup(winName,win)) {
      win = DPWindowC(winName);
      windows[winName] = win;
    }
    hold.Unlock();
    
    // Setup a new port to the window and return it.
    DPWindowOPortC port;
    bool accum = false;
    if(device == "XA") 
      accum = true;
    else {
      RavlAssert(device == "X");
    }
    return DPWindowOPortC(win,accum);
  }


  DPWindowFormatC initDPWindowFormat;
 
}