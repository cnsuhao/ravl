// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/IO.i"
%include "Ravl/Swig2/Image.i"
%include "Ravl/Swig2/Classifier.i"
%include "Ravl/Swig2/ClassifierPreprocess.i"

%{
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/String.hh"
#include "Ravl/PatternRec/Classifier.hh"
#include "Ravl/PatternRec/Function.hh"
%}

namespace RavlN
{
	
  bool Save(const StringC & filename, const RavlImageN::ImageC<ByteT> &);
  bool Load(const StringC & filename, RavlImageN::ImageC<ByteT> &);
  bool Save(const StringC & filename, const RavlImageN::ImageC<RealT> &);
  bool Load(const StringC & filename, RavlImageN::ImageC<RealT> &);
  bool Save(const StringC & filename, const RavlImageN::ImageC<ByteRGBValueC> &);
  bool Load(const StringC & filename, RavlImageN::ImageC<RavlImageN::ByteRGBValueC> &);
  
 
  bool Load(const StringC & filename, ClassifierC & classifier);
  bool Save(const StringC & filename, const ClassifierC & classifier);
  
  bool Load(const StringC & filename, FunctionC & classifier);
  bool Save(const StringC & filename, const FunctionC & classifier);
 
  

}
