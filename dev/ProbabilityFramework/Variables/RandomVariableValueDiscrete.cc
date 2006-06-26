// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValueDiscrete.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableValueDiscreteBodyC::RandomVariableValueDiscreteBodyC(const VariableDiscreteC& variable, const StringC& value)
    : RandomVariableValueBodyC(variable)
  {
    SetValue(value);
  }

  RandomVariableValueDiscreteBodyC::RandomVariableValueDiscreteBodyC(const VariableDiscreteC& variable)
    : RandomVariableValueBodyC(variable)
  {
    //: NOTE that the value hasn't been initialized, you better know what you are doing!
  }

  RandomVariableValueDiscreteBodyC::RandomVariableValueDiscreteBodyC(istream &in)
    : RandomVariableValueBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueDiscreteBodyC(istream &), Unrecognised version number in stream.");
    StringC value;
    in >> value;
    SetValue(value);
  }

  RandomVariableValueDiscreteBodyC::RandomVariableValueDiscreteBodyC(BinIStreamC &in)
    : RandomVariableValueBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueDiscreteBodyC(BinIStream &), Unrecognised version number in stream.");
    StringC value;
    in >> value;
    SetValue(value);
  }
  
  bool RandomVariableValueDiscreteBodyC::Save (ostream &out) const {
    if(!RandomVariableValueBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Value();
    return true;
  }
  
  bool RandomVariableValueDiscreteBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableValueBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Value();
    return true;
  }

  RandomVariableValueDiscreteBodyC::~RandomVariableValueDiscreteBodyC() {
  }
  
  StringC RandomVariableValueDiscreteBodyC::ToString() const {
    return Value();
  }

  const StringC& RandomVariableValueDiscreteBodyC::Value() const {
    return m_value;
  }

  void RandomVariableValueDiscreteBodyC::SetValue(const StringC& value) {
    if (!VariableDiscrete().Values().Contains(value))
      throw ExceptionC("RandomVariableValueDiscreteBodyC::SetValue(), illegal value");
    m_value = value;
  }

  IndexC RandomVariableValueDiscreteBodyC::Index() const {
    return VariableDiscrete().Index(Value());
  }

  bool RandomVariableValueDiscreteBodyC::operator==(const RandomVariableValueC& other) const {
    if (!RandomVariableValueBodyC::operator==(other))
      return false;
    RandomVariableValueDiscreteC otherDiscrete(other);
    if (!otherDiscrete.IsValid())
      return false;
    return Value() == otherDiscrete.Value();
  }

  UIntT RandomVariableValueDiscreteBodyC::Hash() const {
    return RandomVariableValueBodyC::Hash() + Value().Hash();
  }

  VariableDiscreteC RandomVariableValueDiscreteBodyC::VariableDiscrete() const {
    return VariableDiscreteC(Variable());
  }

  static TypeNameC type1(typeid(RandomVariableValueDiscreteC),"RavlProbN::RandomVariableValueDiscreteC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableValueDiscreteBodyC,RandomVariableValueDiscreteC,RandomVariableValueC);
  
}
