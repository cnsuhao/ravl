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

#include "Ravl/PatternRec/DesignClassifierNeuralNetwork2.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class DesignClassifierNeuralNetwork2C : public DesignClassifierSupervisedC {
public:

	DesignClassifierNeuralNetwork2C(UIntT nLayers, UIntT nHidden, bool doNormalisation, RealT regularisation = 0, RealT desiredError = 0.00001,UIntT maxEpochs = 5000,UIntT displayEpochs = 0,bool useSigmoid = true);
	// Construct
	  
	
	
  };
}
