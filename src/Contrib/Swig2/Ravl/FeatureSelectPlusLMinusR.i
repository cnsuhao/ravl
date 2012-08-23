// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/FeatureSelector.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/FeatureSelectPlusLMinusR.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
  
  class FeatureSelectPlusLMinusRC : public FeatureSelectorC
  {
  public:
  
    FeatureSelectPlusLMinusRC(UIntT l, UIntT r, RealT deltaError = 0.001, UIntT numFeatures = 25, UIntT numThreads = 8);
    //: Constructor
    //!param: l      - How many steps forward. l must be bigger than r
    //!param: r      - How many steps backward.
    //!param: numFeatures - determines how many features will be found to build the classifier.
    
    //FeatureSelectPlusLMinusRC(const XMLFactoryContextC &factory);
    //: Construct from XML factory

  };
 
}
