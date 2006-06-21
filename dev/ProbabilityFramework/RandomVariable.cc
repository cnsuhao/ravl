// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/RandomVariable.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableBodyC::RandomVariableBodyC(const StringC& name) {
    SetName(name);
  }

  RandomVariableBodyC::RandomVariableBodyC(istream &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableBodyC(istream &), Unrecognised version number in stream.");
    StringC name(in);
    SetName(name);
  }

  RandomVariableBodyC::RandomVariableBodyC(BinIStreamC &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableBodyC(BinIStream &), Unrecognised version number in stream.");
    StringC name;
    in >> name;
    SetName(name);
  }
  
  bool RandomVariableBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Name();
    return true;
  }
  
  bool RandomVariableBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << Name();
    return true;
  }

  RandomVariableBodyC::~RandomVariableBodyC() {
  }

  const StringC& RandomVariableBodyC::Name() const {
    return m_name;
  }

  void RandomVariableBodyC::SetName(const StringC& name) {
    m_name = downcase(name);
    m_name[0] = toupper(m_name[0]);
  }

  bool RandomVariableBodyC::operator==(const RandomVariableC& other) const {
    return Name() == other.Name();
  }

  UIntT RandomVariableBodyC::Hash() const {
    return Name().Hash();
  }

  ostream &operator<<(ostream &s,const RandomVariableC &obj) {
    obj.Save(s);
    return s;
  }
  
  istream &operator>>(istream &s,RandomVariableC &obj) {
    obj = RandomVariableC(s);
    return s;
  }

  BinOStreamC &operator<<(BinOStreamC &s,const RandomVariableC &obj) {
    obj.Save(s);
    return s;
  }
    
  BinIStreamC &operator>>(BinIStreamC &s,RandomVariableC &obj) {
    obj = RandomVariableC(s);
    return s;
  }
 
  static TypeNameC type1(typeid(RandomVariableC),"RavlProbN::RandomVariableC");

  RAVL_VIRTUALCONSTRUCTOR_HANDLE(RandomVariableBodyC,RandomVariableC,RCHandleVC<RandomVariableBodyC>);
  
}
