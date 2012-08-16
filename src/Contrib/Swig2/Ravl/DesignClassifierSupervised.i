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

#include "Ravl/PatternRec/DesignClassifierSupervised.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class DesignClassifierSupervisedC {
public:
	
	//DesignClassifierSupervisedC();
    
    //DesignClassifierSupervisedC(const XMLFactoryContextC &factory);
    //: Construct from XML factory

    ClassifierC Apply(const SampleC<VectorC> &data,const SampleC<UIntT> &out);
    //: Create a classifier.
    
    ClassifierC Apply(const SampleC<VectorC> &data,const SampleC<UIntT> &out,const SampleC<RealT> &weight);
    //: Create a classifier with weights for the samples.

    ClassifierC Apply(const SampleC<VectorC> &data,const SampleC<UIntT> &out,const SArray1dC<IndexC> &featureSet);
    //: Create a classifier using feature subset
    //!param: in         - sample set of feature vectors
    //!param: out        - sample set of labels
    //!param: featureSet - array of feature indexes to use from sample set when designing classifier
    
    ClassifierC Apply(const SampleC<VectorC> &data,const SampleC<UIntT> &out,const SArray1dC<IndexC> &featureSet,const SampleC<RealT> &weight);
    //: Create a classifier using feature subset
    //!param: in         - sample set of feature vectors
    //!param: out        - sample set of labels
    //!param: featureSet - array of feature indexes to use from sample set when designing classifier
    //!param: weight     - weight associated with each feature vector
    
    ClassifierC Apply(const DataSetVectorLabelC & dset);
    //: Create a classifier from a data set

  };
}
