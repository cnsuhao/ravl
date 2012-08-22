// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/Error.i"
%include "typemaps.i"

%apply double & OUTPUT { double & threshold };

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/ErrorBinaryClassifier.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class ErrorBinaryClassifierC : public ErrorC {
public:
	ErrorBinaryClassifierC();
	// Default constructor
	  
	 RealT FalseRejectRate(const ClassifierC & classifier, const DataSetVectorLabelC & dset, double falseAcceptRate, double & threshold);
    //: Return the false reject rate at a given false accept rate.  This assumes label 0 is the positive class.

    RealT FalseAcceptRate(const ClassifierC & classifier, const DataSetVectorLabelC & dset, double falseRejectRate, double & threshold);
    //: Return the false accept rate at a given false reject rate.  This assumes label 0 is the positive class.
	
	
	
  };
  
}


