// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/VariablePropositionContinuous.hh"
#include "Ravl/StdHash.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  VariablePropositionContinuousBodyC::VariablePropositionContinuousBodyC(const VariableContinuousC& variable, RealT value)
    : VariablePropositionBodyC(variable)
  {
    SetValue(value);
  }

  VariablePropositionContinuousBodyC::VariablePropositionContinuousBodyC(istream &in)
    : VariablePropositionBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("VariablePropositionContinuousBodyC(istream &), Unrecognised version number in stream.");
    RealT value;
    in >> value;
    SetValue(value);
  }

  VariablePropositionContinuousBodyC::VariablePropositionContinuousBodyC(BinIStreamC &in)
    : VariablePropositionBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("VariablePropositionContinuousBodyC(BinIStream &), Unrecognised version number in stream.");
    RealT value;
    in >> value;
    SetValue(value);
  }
  
  bool VariablePropositionContinuousBodyC::Save (ostream &out) const {
    if(!VariablePropositionBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Value();
    return true;
  }
  
  bool VariablePropositionContinuousBodyC::Save (BinOStreamC &out) const {
    if(!VariablePropositionBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Value();
    return true;
  }

  VariablePropositionContinuousBodyC::~VariablePropositionContinuousBodyC() {
  }
  
  StringC VariablePropositionContinuousBodyC::ToString() const {
    return StringC(Value());
  }

  RealT VariablePropositionContinuousBodyC::Value() const {
    return m_value;
  }

  void VariablePropositionContinuousBodyC::SetValue(RealT value) {
    if (!VariableContinuous().Interval().Contains(value))
      throw ExceptionC("VariablePropositionContinuousBodyC::SetValue(), illegal value");
    m_value = value;
  }

  bool VariablePropositionContinuousBodyC::operator==(const VariablePropositionC& other) const {
    if (!VariablePropositionBodyC::operator==(other))
      return false;
    VariablePropositionContinuousC otherContinuous(other);
    if (!otherContinuous.IsValid())
      return false;
    return Value() == otherContinuous.Value();
  }

  UIntT VariablePropositionContinuousBodyC::Hash() const {
    RealT value = Value();
    Int64T bitwiseInt = *((Int64T*)&value);
    return VariablePropositionBodyC::Hash() + StdHash(bitwiseInt);
  }

  VariableContinuousC VariablePropositionContinuousBodyC::VariableContinuous() const {
    return VariableContinuousC(Variable());
  }

  static TypeNameC type1(typeid(VariablePropositionContinuousC),"RavlProbN::VariablePropositionContinuousC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(VariablePropositionContinuousBodyC,VariablePropositionContinuousC,VariablePropositionC);
  
}
