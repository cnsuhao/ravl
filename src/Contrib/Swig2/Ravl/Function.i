// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/Vector.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/Function.hh"
#include "Ravl/StrStream.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class FunctionC {
  public:
    
    VectorC Apply(const VectorC & data) const;
    // Apply function to data
    
    UIntT InputSize() const;
    // Get the input size
    
    UIntT OutputSize() const;
    // Get the output size
    
    %extend {
     const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << *self;
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size())); 
      }
	}
  };
}
