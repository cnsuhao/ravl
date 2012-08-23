// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/Types.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/CommonKernels.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

   class KernelFunctionC {
   public:
   	 KernelFunctionC();
   };


  class LinearKernelC : public KernelFunctionC {
public:

    LinearKernelC(RealT scale);
        
  };
  
  class QuadraticKernelC : public KernelFunctionC {
public:

    QuadraticKernelC(RealT scale);
        
  };
  
  class PolynomialKernelC : public KernelFunctionC {
public:

    PolynomialKernelC(RealT power, RealT scale, RealT B);
        
  };
  
  class RBFKernelC : public KernelFunctionC {
public:

    RBFKernelC(RealT gamma);
        
  };
  
   class Chi2KernelC : public KernelFunctionC {
public:

    Chi2KernelC(RealT gamma);
        
  };
  
  
}
