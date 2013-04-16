// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/Vector.i"
%include "Ravl/Swig2/IO.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/Classifier.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class ClassifierC {
public:
	ClassifierC();
	// Empty classifier
	  
	UIntT Classify(const VectorC & vector) const;
	// Classify vector
	
	VectorC Confidence(const VectorC & vector) const;
	// Estimate confidence of each label
	
	
  };

}


