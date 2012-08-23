// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DesignClassifierSupervised.i"
%include "Ravl/Swig2/DataSetVectorLabel.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/FeatureSelector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
  
  class FeatureSelectorC 
  {
public:
  FeatureSelectorC();
 
  SArray1dC<IndexC> SelectFeatures(DesignClassifierSupervisedC &designer,
				     const DataSetVectorLabelC &train, 
				     const DataSetVectorLabelC &test,
				     ClassifierC &classifier) const;
    //: Select feature subset which gives optimal performance
    // This is evaluated using a given 
    //!param: designer - supervised classifier designer used to create classifiers
    //!param: train - training dataset with feature vectors and labels
    //!param: test - testing dataset with feature vectors and labels
    //!param: classifier - final classifier which gave best performance with optimised feature set
    //!return: an array containing the feature indexes that are in the selected set
  };
 
}
