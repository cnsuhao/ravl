// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%{
#include "Ravl/StrStream.hh"
%}

%define __STR__()
const char *__str__() {
      RavlN::StrOStreamC os;
      os << *self;
      return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size())); 
    }	    
%enddef

%define __NONZERO__()
bool __nonzero__() {
		return self->IsValid();
    }	    
%enddef

