// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DesignFunctionUnsupervised.i"
%include "Ravl/Swig2/Function.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DesignFuncReduce.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
	class DesignFuncReduceC
    : public DesignFunctionUnsupervisedC
  {
  public:
    DesignFuncReduceC();
    //: Default constructor.
    // Creates an invalid handle.
    
    RealT VariationPreserved() const; 
    //: Returns amount of variation to attempt to preserve in reduced set.
    // 0 -> None; 1 -> All; >1 (truncated to int) -> Size of set preserved.
  };


}
