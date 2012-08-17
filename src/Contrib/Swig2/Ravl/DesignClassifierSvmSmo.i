// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DesignSvm.i"
%include "Ravl/Swig2/CommonKernels.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DesignClassifierSvmSmo.hh"
#include "Ravl/PatternRec/CommonKernels.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class DesignSvmSmoC : public DesignSvmC {
public:

   DesignSvmSmoC(const KernelFunctionC & KernelFunction, RealT Penalty1 = 1000, RealT Penalty2 = 1000, RealT Tolerance = 0.0000001, RealT Eps = 0.000000001, RealT LambdaThreshold = 0.000000000001); 
      
  };
}
