// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/Macros.i"
%include "Ravl/Swig2/VectorMatrix.i"
%include "Ravl/Swig2/Sample.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/Function.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class FunctionC {
  public:
    
    VectorC Apply(const VectorC &data) const;
    //: Apply function to 'data'
    
    VectorC Apply(const VectorC & data1, const VectorC & data2) const;
    //: Apply function to two data vectors 
    // The default behaviour is to concatenate the two vectors  
    // and then call the single vector version of Apply() 
    
    SampleC<VectorC> Apply(const SampleC<VectorC> &data);
    //: Apply transform to whole dataset.
    
    VectorC operator() (const VectorC &X) const; 
    //: Evaluate Y=f(X)
    
    bool CheckJacobian(const VectorC &X,RealT tolerance = 1e-4,RealT epsilon = 1e-4) const;
    //: Compare the numerical and computed jacobians at X, return true if the match.
    // Useful for debugging!

    MatrixC Jacobian(const VectorC &X) const;
    //: Calculate Jacobian matrix at X
    // Performs numerical estimation of the Jacobian using differences. This
    // function has and should be overloaded for all cases where the Jacobian
    // can be calculated analytically.
    
    VectorMatrixC Evaluate(const VectorC &X);
    //: Evaluate the function and its jacobian at the same time.
    // This method defaults to calling 'Apply' and 'Jacobian' sperately.
    
    bool EvaluateValueJacobian(const VectorC &X,VectorC &value,MatrixC &jacobian) const;
    //: Evaluate the value,jacobian of the function at point X
    // Returns true if all values are provide, false if one or more is approximated.
    
    bool EvaluateValue(const VectorC &X,VectorC &value) const;
    //: Evaluate the value of the function at point X
    // Returns true if all values are provide, false if one or more is approximated.
    
    UIntT InputSize() const;
    //: Size of input vector
    
    UIntT OutputSize() const;
    //: Size of output vector

       
	%extend {
		__STR__();
		__NONZERO__();
	}
	
	
  };
}
