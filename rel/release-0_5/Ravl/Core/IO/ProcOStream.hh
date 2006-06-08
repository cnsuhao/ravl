// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLDPPROCOSTREAM_HEADER
#define RAVLDPPROCOSTREAM_HEADER 1
/////////////////////////////////////////////////////
//! example=exDataProc.cc
//! rcsid="$Id$"
//! file="Ravl/Core/IO/ProcOStream.hh"
//! lib=RavlIO
//! author="Charles Galambos"
//! docentry="Ravl.Core.Data Processing" 
//! date="06/07/98"
//! userlevel=Default

#include "Ravl/DP/StreamOp.hh"
#include "Ravl/DP/Process.hh"

namespace RavlN {
  
  ////////////////////////////////////////
  //! userlevel=Develop
  //: Wrapped process body.
  
  template<class InT,class OutT>
  class DPProcOStreamBodyC 
    : public DPProcessC<InT,OutT>,
      public DPOStreamOpBodyC<InT,OutT>
  {
  public:
    DPProcOStreamBodyC(const DPProcessC<InT,OutT> &bod,const DPOPortC<OutT> &nout)
      : DPProcessC<InT,OutT>(bod),
      DPOStreamOpBodyC<InT,OutT>(nout)
      {}
    //: Constructor.
    
    virtual bool Put(const InT &dat) { 
      RavlAssert(output.IsValid());
      return output.Put(Apply(dat)); 
    }
    //: Process next piece of data.  
    
    virtual IntT PutArray(SArray1dC<InT> &src) {
      SArray1dC<OutT> dest(src.Size());
#ifdef NDEBUG
      ApplyArray(src,dest);
#else
      RavlAssert((UIntT) ApplyArray(src,dest) == src.Size());
#endif
      return output.PutArray(dest);
    }
    //: Get Array of data.
    
    
  }; 

  /////////////////////////////////
  //! userlevel=Normal
  //: Wrapped Proccess handle.
  
  template<class InT,class OutT>
  class DPProcOStreamC 
    : public DPOStreamOpC<InT,OutT> 
  {
  public:
    DPProcOStreamC() 
      : DPEntityC(true)
      {}
    //: Default Constructor.
    // Creates an invalid handle.
    
    DPProcOStreamC(const DPProcessC<InT,OutT> &bod,const DPOPortC<OutT> &nout)
      : DPEntityC((DPOPortBodyC<InT> &) *new DPProcOStreamBodyC<InT,OutT>(bod,nout))
      {}
    //: Constructor.
    // 
    
    DPProcOStreamC(const DPProcOStreamC<IntT,OutT> &oth) 
      : DPEntityC(oth),
      DPOStreamOpC<InT,OutT>(oth)
      {}
    //: Copy Constructor.
    
  };
  
  ///////////////////////////////
  //: Composition operator.
  
  //template<class InT,class OutT>
  //DPProcOStreamC<InT,OutT> operator>>(const DPIPortC<InT> &in,const DPProcessC<InT,OutT> &proc) 
  //{ return DPProcOStreamC<InT,OutT> (proc,in); }
  
}

#endif