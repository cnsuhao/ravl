// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DesignFuncReduce.i"
%include "Ravl/Swig2/VectorMatrix.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DesignFuncPCA.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

	 class DesignFuncPCAC
    : public DesignFuncReduceC
  {
  public:

    DesignFuncPCAC();
    //: Default constructor.
    // Creates an invalid handle.
    
    DesignFuncPCAC(RealT variationPreserved);
    //: Constructor 
    // "variationPreserved" is amount of variation to attempt to preserve in reduced set:<br>
    // 0 -> None; 1 -> All; >1 (truncated to int) -> Size of set preserved.
    
    VectorMatrixC &Pca();
    //: Access eigen vectors and values.

    const VectorMatrixC &Pca() const;
    //: Access eigen vectors and values.
    
    VectorC &Mean();
    //: Access mean of input vectors.
    
    const VectorC &Mean() const;
    //: Access mean of input vectors.
  };


}
