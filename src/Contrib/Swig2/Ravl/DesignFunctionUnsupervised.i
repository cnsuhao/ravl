// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/Classifier.i"
%include "Ravl/Swig2/DataSetVectorLabel.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DesignFunctionUnsupervised.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class DesignFunctionUnsupervisedC
  {
  public:
    DesignFunctionUnsupervisedC();
    //: Default constructor.
    // Creates an invalid handle.
    
    FunctionC Apply(const SampleC<VectorC> & data);
    //: Create function from the given data.
    
    FunctionC Apply(const SampleC<VectorC> & data, const SampleC<RealT> &weight);
    //: Create function from the given data, and sample weights.
    
    //FunctionC Apply(SampleStreamC<VectorC> & data);
    //: Create function from the given data.
    // Note: Construction from a sample stream may not be implemented by all designers.
    
    //FunctionC Apply(SampleStream2C<VectorC,RealT> & data);
    //: Create function from the given data, and sample weights.
    // Note: Construction from a sample stream may not be implemented by all designers.

    FunctionC Apply(const SampleC<TVectorC<float> > & data);
    //: Create function from the given data.

    FunctionC Apply(const SampleC<TVectorC<float> > & data,const SampleC<float> &weight);
    //: Create function from the given data, and sample weights.

    //FunctionC Apply(SampleStreamC<TVectorC<float> > & data);
    //: Create function from the given data.
    // Note: Construction from a sample stream may not be implemented by all designers.
    
    //FunctionC Apply(SampleStream2C<TVectorC<float>,float> & data);
    //: Create function from the given data, and sample weights.
    // Note: Construction from a sample stream may not be implemented by all designers.

  };

}
