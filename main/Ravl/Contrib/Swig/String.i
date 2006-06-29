// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig/Types.i"

%{
#include "Ravl/String.hh"
%}

namespace RavlN {
  
  class StringC {
  public:
    StringC();
    StringC(const char *);
    
    bool IsEmpty() const;
    const char *chars() const;
    %extend {
      const char *__str__() 
	{ return self->chars(); }
    }
  };
  
}
