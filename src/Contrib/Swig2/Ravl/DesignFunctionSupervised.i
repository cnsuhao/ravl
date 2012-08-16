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

#include "Ravl/PatternRec/DesignFunctionSupervised.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class DesignFunctionSupervisedC {
public:
	
	 DesignFunctionSupervisedC();
    //: Default constructor.
    // Creates an invalid handle.

    DesignFunctionSupervisedC(const DesignFunctionSupervisedC &other);
    //: Copy constructor.

    //DesignFunctionSupervisedC(const XMLFactoryContextC &factory);
    //: Construct from XML factory
    
    FunctionC Apply(const SampleC<VectorC> &data,const SampleC<VectorC> &out);
    //: Create function from the given data.
    
    FunctionC Apply(const SampleC<VectorC> &data,const SampleC<VectorC> &out,const SampleC<RealT> &weight);
    //: Create function from the given data, and sample weights.
    
    //FunctionC Apply(SampleStream2C<VectorC,VectorC > &data);
    //: Create function from the given data (in,out)
    // Note: Construction from a sample stream may not be implemented by all designers.
    
    //FunctionC Apply(SampleStream3C<VectorC,VectorC,RealT> &data);
    //: Create function from the given data, and sample weights. (in,out,weight)
    // Note: Construction from a sample stream may not be implemented by all designers.

    FunctionC Apply(const SampleC<TVectorC<float> > &data,const SampleC<TVectorC<float> > &out);
    //: Create function from the given data.

    FunctionC Apply(const SampleC<TVectorC<float> > &data,const SampleC<TVectorC<float> > &out,const SampleC<float> &weight);
    //: Create function from the given data, and sample weights.

    //FunctionC Apply(SampleStream2C<TVectorC<float>,TVectorC<float> > &in);
    //: Create function from the given data.
    // Note: Construction from a sample stream may not be implemented by all designers.
    
    //FunctionC Apply(SampleStream3C<TVectorC<float>,TVectorC<float>,float> &in);
    //: Create function from the given data, and sample weights.
    // Note: Construction from a sample stream may not be implemented by all designers.

  };
}
