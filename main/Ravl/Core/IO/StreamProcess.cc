//////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! date="16/7/2002"
//! author="Charles Galambos"

#include "Ravl/DP/StreamProcess.hh"
#include "Ravl/HashIter.hh"

namespace RavlN {

  //: Input plugs.
  
  DListC<DPIPlugBaseC> DPStreamProcessBodyC::IPlugs() const {
    DListC<DPIPlugBaseC> ret;
    for(HashIterC<StringC,DPIPlugBaseC> it(iplugs);it;it++)
      ret.InsFirst(it.Data());
    return ret;
  }
  
  //: Output plugs
  
  DListC<DPOPlugBaseC> DPStreamProcessBodyC::OPlugs() const {
    DListC<DPOPlugBaseC> ret;
    for(HashIterC<StringC,DPOPlugBaseC> it(oplugs);it;it++)
      ret.InsFirst(it.Data());
    return ret;
  }
  
  //: Input ports.
  
  DListC<DPIPortBaseC> DPStreamProcessBodyC::IPorts() const {
    DListC<DPIPortBaseC> ret;
    for(HashIterC<StringC,DPIPortBaseC> it(iports);it;it++)
      ret.InsFirst(it.Data());
    return ret;
  }
  
  //: Output ports
  
  DListC<DPOPortBaseC> DPStreamProcessBodyC::OPorts() const {
    DListC<DPOPortBaseC> ret;
    for(HashIterC<StringC,DPOPortBaseC> it(oports);it;it++)
      ret.InsFirst(it.Data());
    return ret;
  }

}
