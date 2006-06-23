// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValue.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableValueBodyC::RandomVariableValueBodyC(const RandomVariableC& variable) {
    SetRandomVariable(variable);
  }

  RandomVariableValueBodyC::RandomVariableValueBodyC(istream &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueBodyC(istream &), Unrecognised version number in stream.");
    RandomVariableC variable(in);
    SetRandomVariable(variable);
  }

  RandomVariableValueBodyC::RandomVariableValueBodyC(BinIStreamC &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableValueBodyC(BinIStream &), Unrecognised version number in stream.");
    RandomVariableC variable(in);
    SetRandomVariable(variable);
  }
  
  bool RandomVariableValueBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << RandomVariable();
    return true;
  }
  
  bool RandomVariableValueBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << RandomVariable();
    return true;
  }

  RandomVariableValueBodyC::~RandomVariableValueBodyC() {
  }

  const RandomVariableC& RandomVariableValueBodyC::RandomVariable() const {
    return m_variable;
  }

  void RandomVariableValueBodyC::SetRandomVariable(const RandomVariableC& variable) {
    if (!variable.IsValid())
      throw ExceptionC("RandomVariableValueBodyC::SetRandomVariable() with invalid variable");
    m_variable = variable;
  }

  bool RandomVariableValueBodyC::operator==(const RandomVariableValueC& other) const {
    return RandomVariable() == other.RandomVariable();
  }

  UIntT RandomVariableValueBodyC::Hash() const {
    return RandomVariable().Hash();
  }

  ostream &operator<<(ostream &s,const RandomVariableValueC &obj) {
    obj.Save(s);
    return s;
  }
  
  istream &operator>>(istream &s,RandomVariableValueC &obj) {
    obj = RandomVariableValueC(s);
    return s;
  }

  BinOStreamC &operator<<(BinOStreamC &s,const RandomVariableValueC &obj) {
    obj.Save(s);
    return s;
  }
    
  BinIStreamC &operator>>(BinIStreamC &s,RandomVariableValueC &obj) {
    obj = RandomVariableValueC(s);
    return s;
  }
 
  static TypeNameC type1(typeid(RandomVariableValueC),"RavlProbN::RandomVariableValueC");
    
  RAVL_VIRTUALCONSTRUCTOR_HANDLE(RandomVariableValueBodyC,RandomVariableValueC,RCHandleVC<RandomVariableValueBodyC>);
  
}
