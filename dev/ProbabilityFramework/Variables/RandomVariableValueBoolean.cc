// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValueBoolean.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableValueBooleanBodyC::RandomVariableValueBooleanBodyC(const VariableBooleanC& variable, bool value)
    : RandomVariableValueDiscreteBodyC(variable)
  {
    SetBooleanValue(value);
  }

  RandomVariableValueBooleanBodyC::RandomVariableValueBooleanBodyC(istream &in)
    : RandomVariableValueDiscreteBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueBooleanBodyC(istream &), Unrecognised version number in stream.");
    bool value;
    in >> value;
    SetBooleanValue(value);
  }

  RandomVariableValueBooleanBodyC::RandomVariableValueBooleanBodyC(BinIStreamC &in)
    : RandomVariableValueDiscreteBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueBooleanBodyC(BinIStream &), Unrecognised version number in stream.");
    bool value;
    in >> value;
    SetBooleanValue(value);
  }
  
  bool RandomVariableValueBooleanBodyC::Save (ostream &out) const {
    if(!RandomVariableValueDiscreteBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << BooleanValue();
    return true;
  }
  
  bool RandomVariableValueBooleanBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableValueDiscreteBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << BooleanValue();
    return true;
  }

  RandomVariableValueBooleanBodyC::~RandomVariableValueBooleanBodyC() {
  }
  
  bool RandomVariableValueBooleanBodyC::BooleanValue() const {
    return m_booleanValue;
  }

  void RandomVariableValueBooleanBodyC::SetBooleanValue(bool value) {
    m_booleanValue = value;
    RandomVariableValueDiscreteBodyC::SetValue(VariableBoolean().Value(value));
  }

  void RandomVariableValueBooleanBodyC::SetValue(const StringC& value) {
    SetBooleanValue(VariableBoolean().Value(true) == value);
  }

  VariableBooleanC RandomVariableValueBooleanBodyC::VariableBoolean() const {
    return VariableBooleanC(Variable());
  }

  static TypeNameC type1(typeid(RandomVariableValueBooleanC),"RavlProbN::RandomVariableValueBooleanC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableValueBooleanBodyC,RandomVariableValueBooleanC,RandomVariableValueDiscreteC);
  
}
