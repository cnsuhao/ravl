// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

// Set the Python module 'docstring'
%define RAVL_DOCSTRING
"Recognition and Vision Library
RAVL provides a base C++ class library together with a range of
computer vision, pattern recognition and supporting tools."
%enddef

%module(docstring=RAVL_DOCSTRING) Ravl

// Enable basic Python automatic 'docstring' entries for  
// function arguments and return values
%feature("autodoc", "0");
  
%include "Ravl/Swig2/RavlCore.i"
%include "Ravl/Swig2/RavlPatternRec.i"
%include "Ravl/Swig2/RavlContrib.i"

