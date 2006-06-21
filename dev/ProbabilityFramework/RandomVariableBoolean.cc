// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableBoolean.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableBooleanBodyC::RandomVariableBooleanBodyC(const StringC& name)
    : RandomVariableDiscreteBodyC(name)
  {
    SetValueNames();
    HSetC<StringC> values;
    values.Insert(m_trueValue);
    values.Insert(m_falseValue);
    SetValues(values);
  }

  RandomVariableBooleanBodyC::RandomVariableBooleanBodyC(istream &in)
    : RandomVariableDiscreteBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableBooleanBodyC(istream &), Unrecognised version number in stream.");
    SetValueNames();
  }

  RandomVariableBooleanBodyC::RandomVariableBooleanBodyC(BinIStreamC &in)
    : RandomVariableDiscreteBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableBooleanBodyC(BinIStream &), Unrecognised version number in stream.");
    SetValueNames();
  }
  
  bool RandomVariableBooleanBodyC::Save (ostream &out) const {
    if(!RandomVariableDiscreteBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version;
    return true;
  }
  
  bool RandomVariableBooleanBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableDiscreteBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version;
    return true;
  }

  RandomVariableBooleanBodyC::~RandomVariableBooleanBodyC() {
  }

  const StringC& RandomVariableBooleanBodyC::Value(bool value) const {
    return value? m_trueValue: m_falseValue;
  }
  
  void RandomVariableBooleanBodyC::SetValueNames() {
    m_trueValue = downcase(Name());
    m_falseValue = StringC("¬")+downcase(Name());
  }

  static TypeNameC type1(typeid(RandomVariableBooleanC),"RavlProbN::RandomVariableBooleanC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableBooleanBodyC,RandomVariableBooleanC,RandomVariableDiscreteC);
  
}
