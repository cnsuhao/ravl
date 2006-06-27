// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/Proposition.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PropositionBodyC::PropositionBodyC(const VariableSetC& variableSet, const HSetC<VariablePropositionC>& values) {
    SetVariableSet(variableSet);
    SetValues(values);
  }

  PropositionBodyC::PropositionBodyC(const PropositionBodyC& other, const VariablePropositionC& value) {
    SetVariableSet(other.VariableSet());
    HSetC<VariablePropositionC> values = other.Values().Copy();
    values.Insert(value);
    SetValues(values);
  }

  PropositionBodyC::PropositionBodyC(istream &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("PropositionBodyC(istream &), Unrecognised version number in stream.");
    VariableSetC variableSet(in);
    SetVariableSet(variableSet);
    HSetC<VariablePropositionC> values;
    in >> values;
    SetValues(values);
  }

  PropositionBodyC::PropositionBodyC(BinIStreamC &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("PropositionBodyC(BinIStream &), Unrecognised version number in stream.");
    VariableSetC variableSet(in);
    SetVariableSet(variableSet);
    HSetC<VariablePropositionC> values;
    in >> values;
    SetValues(values);
  }
  
  bool PropositionBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << VariableSet() << ' ' << Values();
    return true;
  }
  
  bool PropositionBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << VariableSet() << Values();
    return true;
  }

  PropositionBodyC::~PropositionBodyC() {
  }

  const VariableSetC& PropositionBodyC::VariableSet() const {
    return m_variableSet;
  }

  SizeT PropositionBodyC::NumValues() const {
    return Values().Size();
  }

  const HSetC<VariablePropositionC>& PropositionBodyC::Values() const {
    return m_values;
  }

  const VariablePropositionC& PropositionBodyC::Value(IndexC index) const {
    if (index < 0 || index >= NumValues())
      throw ExceptionC("PropositionBodyC::Value(), index out of range");
    HSetIterC<VariablePropositionC> it(Values());
    while(index--)
      it++;
    return *it;
  }

  StringC PropositionBodyC::ToString() const {
    HSetIterC<VariablePropositionC> it(Values());
    StringC string = it->Variable().Name() + "=" + it->ToString();
    for (it++; it; it++) {
      string += ",";
      string += it->Variable().Name() + "=" + it->ToString();
    }
    return string;
  }

  StringC PropositionBodyC::LotteryName() const {
    StringC name = VariableSet().ToString(); // show all variables
    HSetC<VariableC> variableSetSet = VariableSet().Variables().Copy();
    for (HSetIterC<VariablePropositionC> it(Values()); it; it++) {
      variableSetSet.Remove(it->Variable());
    }
    name += "->(";
    if (variableSetSet.Size() > 0) {
      HSetIterC<VariableC> it(variableSetSet);
      name += it->Name();
      for (it++; it; it++) {
        name += ",";
        name += it->Name();
      }
    }
    name += ")";
    return name;
  }

  PropositionC PropositionBodyC::SubProposition(const VariableSetC& subVariableSet) const {
  	// Check that all variables are in current variableSet!
  	for (HSetIterC<VariableC> dt(subVariableSet.Variables()); dt; dt++)
  		if (!VariableSet().Contains(*dt))
  			throw ExceptionC("PropositionBodyC::SubProposition(), invalid new variableSet variable");
    HSetC<VariablePropositionC> values;
    for (HSetIterC<VariablePropositionC> ht(Values()); ht; ht++) {
      if (subVariableSet.Variables().Contains(ht->Variable()))
        values.Insert(*ht);
    }
    return PropositionC(subVariableSet, values);
  }

  void PropositionBodyC::SetVariableSet(const VariableSetC& variableSet) {
    m_variableSet = variableSet;
  }

  void PropositionBodyC::SetValues(const HSetC<VariablePropositionC>& values) {
    //:FIXME- what collection for efficiency?
    for (HSetIterC<VariablePropositionC> it(values); it; it++)
      if (!VariableSet().Contains(it->Variable()))
        throw ExceptionC("PropositionBodyC::SetValues(), value not in variableSet");
    m_values = values.Copy();
  }

  bool PropositionBodyC::operator==(const PropositionBodyC& other) const {
    if (this == &other)
      return true;
    if (VariableSet() != other.VariableSet())
      return false;
    if (NumValues() != other.NumValues())
      return false;
    for (HSetIterC<VariablePropositionC> ht(Values()); ht; ht++)
      if (!other.Values().Contains(*ht))
        return false;
    return true;
    
  }

  UIntT PropositionBodyC::Hash() const {
    UIntT hash = 0;
    for (HSetIterC<VariablePropositionC> it(Values()); it; it++)
      hash += it->Hash();
    return VariableSet().Hash() + hash;
  }

  ostream &operator<<(ostream &s,const PropositionC &obj) {
    obj.Save(s);
    return s;
  }
  
  istream &operator>>(istream &s,PropositionC &obj) {
    obj = PropositionC(s);
    return s;
  }

  BinOStreamC &operator<<(BinOStreamC &s,const PropositionC &obj) {
    obj.Save(s);
    return s;
  }
    
  BinIStreamC &operator>>(BinIStreamC &s,PropositionC &obj) {
    obj = PropositionC(s);
    return s;
  }
 
  static TypeNameC type1(typeid(PropositionC),"RavlProbN::PropositionC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(PropositionBodyC,PropositionC,RCHandleVC<PropositionBodyC>);
  
}
