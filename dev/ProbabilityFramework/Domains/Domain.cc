// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/Domain.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  DomainBodyC::DomainBodyC(const HSetC<VariableC>& variables) {
    SetVariables(variables);
  }

  DomainBodyC::DomainBodyC(istream &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("DomainBodyC(istream &), Unrecognised version number in stream.");
    HSetC<VariableC> variables;
    in >> variables;
    SetVariables(variables);
  }

  DomainBodyC::DomainBodyC(BinIStreamC &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("DomainBodyC(BinIStream &), Unrecognised version number in stream.");
    HSetC<VariableC> variables;
    in >> variables;
    SetVariables(variables);
  }
  
  bool DomainBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Variables();
    return true;
  }
  
  bool DomainBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << Variables();
    return true;
  }

  DomainBodyC::~DomainBodyC() {
  }

  bool DomainBodyC::operator==(const DomainBodyC& other) const {
    if (this == &other)
      return true;
    if (NumVariables() != other.NumVariables())
      return false;
    for (HSetIterC<VariableC> ht(Variables()); ht; ht++)
      if (!other.Variables().Contains(*ht))
        return false;
    return true;
  }

  bool DomainBodyC::Contains(const VariableC& variable) const {
    //:FIXME- depends on choice of collection
    if (!variable.IsValid())
    	return false;
    return Variables().Contains(variable);
  }

  SizeT DomainBodyC::NumVariables() const {
    return Variables().Size();
  }

  const HSetC<VariableC>& DomainBodyC::Variables() const {
    return m_variables;
  }

  const VariableC& DomainBodyC::Variable(IndexC index) const {
    if (index < 0 || index >= NumVariables())
      throw ExceptionC("DomainBodyC::Variable(), index too big");
    HSetIterC<VariableC> it(Variables());
    while(index--)
      it++;
    return *it;
  }

  void DomainBodyC::SetVariables(const HSetC<VariableC>& variables) {
    //:FIXME- what collection for efficiency?
    m_variables = variables.Copy();
  }

  IndexC DomainBodyC::Index(const VariableC& variable) const {
  	if (!variable.IsValid())
  		throw ExceptionC("DomainBodyC::Index(), invalid variable");
    //:FIXME- this should probably be implemented using hash table
    IndexC index(0);
    for (HSetIterC<VariableC> it(Variables()); it; it++, index++)
      if (*it == variable)
        return index;
    throw ExceptionC("DomainBodyC::Index(), couldn't find variable");
  }

  StringC DomainBodyC::ToString() const {
    HSetIterC<VariableC> it(Variables());
    StringC string = it->Name();
    for (it++; it; it++) {
      string += ",";
      string += it->Name();
    }
    return string;
  }

  UIntT DomainBodyC::Hash() const {
    UIntT hash = 0;
    for (HSetIterC<VariableC> it(Variables()); it; it++)
      hash += it->Hash();
    return hash;
  }

  ostream &operator<<(ostream &s,const DomainC &obj) {
    obj.Save(s);
    return s;
  }
  
  istream &operator>>(istream &s,DomainC &obj) {
    obj = DomainC(s);
    return s;
  }

  BinOStreamC &operator<<(BinOStreamC &s,const DomainC &obj) {
    obj.Save(s);
    return s;
  }
    
  BinIStreamC &operator>>(BinIStreamC &s,DomainC &obj) {
    obj = DomainC(s);
    return s;
  }
 
  static TypeNameC type1(typeid(DomainC),"RavlProbN::DomainC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(DomainBodyC,DomainC,RCHandleVC<DomainBodyC>);
  
}
