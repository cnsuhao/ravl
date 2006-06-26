// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValueContinuous.hh"
#include "Ravl/StdHash.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableValueContinuousBodyC::RandomVariableValueContinuousBodyC(const VariableContinuousC& variable, RealT value)
    : RandomVariableValueBodyC(variable)
  {
    SetValue(value);
  }

  RandomVariableValueContinuousBodyC::RandomVariableValueContinuousBodyC(istream &in)
    : RandomVariableValueBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueContinuousBodyC(istream &), Unrecognised version number in stream.");
    RealT value;
    in >> value;
    SetValue(value);
  }

  RandomVariableValueContinuousBodyC::RandomVariableValueContinuousBodyC(BinIStreamC &in)
    : RandomVariableValueBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueContinuousBodyC(BinIStream &), Unrecognised version number in stream.");
    RealT value;
    in >> value;
    SetValue(value);
  }
  
  bool RandomVariableValueContinuousBodyC::Save (ostream &out) const {
    if(!RandomVariableValueBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Value();
    return true;
  }
  
  bool RandomVariableValueContinuousBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableValueBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Value();
    return true;
  }

  RandomVariableValueContinuousBodyC::~RandomVariableValueContinuousBodyC() {
  }
  
  StringC RandomVariableValueContinuousBodyC::ToString() const {
    return StringC(Value());
  }

  RealT RandomVariableValueContinuousBodyC::Value() const {
    return m_value;
  }

  void RandomVariableValueContinuousBodyC::SetValue(RealT value) {
    if (!VariableContinuous().Interval().Contains(value))
      throw ExceptionC("RandomVariableValueContinuousBodyC::SetValue(), illegal value");
    m_value = value;
  }

  bool RandomVariableValueContinuousBodyC::operator==(const RandomVariableValueC& other) const {
    if (!RandomVariableValueBodyC::operator==(other))
      return false;
    RandomVariableValueContinuousC otherContinuous(other);
    if (!otherContinuous.IsValid())
      return false;
    return Value() == otherContinuous.Value();
  }

  UIntT RandomVariableValueContinuousBodyC::Hash() const {
    RealT value = Value();
    Int64T bitwiseInt = *((Int64T*)&value);
    return RandomVariableValueBodyC::Hash() + StdHash(bitwiseInt);
  }

  VariableContinuousC RandomVariableValueContinuousBodyC::VariableContinuous() const {
    return VariableContinuousC(Variable());
  }

  static TypeNameC type1(typeid(RandomVariableValueContinuousC),"RavlProbN::RandomVariableValueContinuousC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableValueContinuousBodyC,RandomVariableValueContinuousC,RandomVariableValueC);
  
}
