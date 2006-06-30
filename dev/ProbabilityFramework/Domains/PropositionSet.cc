// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/PropositionSet.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PropositionSetBodyC::PropositionSetBodyC(const VariableSetC& variableSet, const HSetC<VariablePropositionC>& values) {
    SetVariableSet(variableSet);
    SetValues(values);
  }

  PropositionSetBodyC::PropositionSetBodyC(const VariableSetC& variableSet, const VariablePropositionC& value) {
    SetVariableSet(variableSet);
    SetSingleValue(value);
  }

  PropositionSetBodyC::PropositionSetBodyC(const VariablePropositionC& value) {
    SetVariableSet(VariableSetC(value.Variable()));
    SetSingleValue(value);
  }

  PropositionSetBodyC::PropositionSetBodyC(const PropositionSetBodyC& other, const VariablePropositionC& value) {
    SetVariableSet(other.VariableSet());
    HSetC<VariablePropositionC> values = other.Values().Copy();
    values.Insert(value);
    SetValues(values);
  }

  PropositionSetBodyC::PropositionSetBodyC(istream &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("PropositionSetBodyC(istream &), Unrecognised version number in stream.");
    VariableSetC variableSet(in);
    SetVariableSet(variableSet);
    HSetC<VariablePropositionC> values;
    in >> values;
    SetValues(values);
  }

  PropositionSetBodyC::PropositionSetBodyC(BinIStreamC &in)
    : RCBodyVC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("PropositionSetBodyC(BinIStream &), Unrecognised version number in stream.");
    VariableSetC variableSet(in);
    SetVariableSet(variableSet);
    HSetC<VariablePropositionC> values;
    in >> values;
    SetValues(values);
  }
  
  bool PropositionSetBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << VariableSet() << ' ' << Values();
    return true;
  }
  
  bool PropositionSetBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << VariableSet() << Values();
    return true;
  }

  PropositionSetBodyC::~PropositionSetBodyC() {
  }

  const VariableSetC& PropositionSetBodyC::VariableSet() const {
    return m_variableSet;
  }

  SizeT PropositionSetBodyC::Size() const {
    return Values().Size();
  }

  const HSetC<VariablePropositionC>& PropositionSetBodyC::Values() const {
    return m_values;
  }

  const VariablePropositionC& PropositionSetBodyC::Value(IndexC index) const {
    if (index < 0 || index >= Size())
      throw ExceptionC("PropositionSetBodyC::Value(), index out of range");
    HSetIterC<VariablePropositionC> it(Values());
    while(index--)
      it++;
    return *it;
  }

  StringC PropositionSetBodyC::ToString() const {
  	if (Size() == 0)
  	  return "";
    HSetIterC<VariablePropositionC> it(Values());
    StringC string = it->Variable().Name() + "=" + it->ToString();
    for (it++; it; it++) {
      string += ",";
      string += it->Variable().Name() + "=" + it->ToString();
    }
    return string;
  }

  StringC PropositionSetBodyC::LotteryName() const {
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

  PropositionSetC PropositionSetBodyC::SubPropositionSet(const VariableSetC& subVariableSet) const {
  	// Check that all variables are in current variableSet!
  	for (HSetIterC<VariableC> dt(subVariableSet.Variables()); dt; dt++)
  		if (!VariableSet().Contains(*dt))
  			throw ExceptionC("PropositionSetBodyC::SubPropositionSet(), invalid new variableSet variable");
    HSetC<VariablePropositionC> values;
    for (HSetIterC<VariablePropositionC> ht(Values()); ht; ht++) {
      if (subVariableSet.Variables().Contains(ht->Variable()))
        values.Insert(*ht);
    }
    return PropositionSetC(subVariableSet, values);
  }

  void PropositionSetBodyC::SetVariableSet(const VariableSetC& variableSet) {
    m_variableSet = variableSet;
  }

  void PropositionSetBodyC::SetValues(const HSetC<VariablePropositionC>& values) {
    //:FIXME- what collection for efficiency?
    for (HSetIterC<VariablePropositionC> it(values); it; it++)
      if (!VariableSet().Contains(it->Variable()))
        throw ExceptionC("PropositionSetBodyC::SetValues(), value not in variableSet");
    m_values = values.Copy();
  }

  void PropositionSetBodyC::SetSingleValue(const VariablePropositionC& value) {
  	HSetC<VariablePropositionC> values;
  	values.Insert(value);
  	SetValues(values);
  }
  
  bool PropositionSetBodyC::operator==(const PropositionSetBodyC& other) const {
    if (this == &other)
      return true;
    if (VariableSet() != other.VariableSet())
      return false;
    if (Size() != other.Size())
      return false;
    for (HSetIterC<VariablePropositionC> ht(Values()); ht; ht++)
      if (!other.Values().Contains(*ht))
        return false;
    return true;
    
  }

  UIntT PropositionSetBodyC::Hash() const {
    UIntT hash = 0;
    for (HSetIterC<VariablePropositionC> it(Values()); it; it++)
      hash += it->Hash();
    return VariableSet().Hash() + hash;
  }

  ostream &operator<<(ostream &s,const PropositionSetC &obj) {
    obj.Save(s);
    return s;
  }
  
  istream &operator>>(istream &s,PropositionSetC &obj) {
    obj = PropositionSetC(s);
    return s;
  }

  BinOStreamC &operator<<(BinOStreamC &s,const PropositionSetC &obj) {
    obj.Save(s);
    return s;
  }
    
  BinIStreamC &operator>>(BinIStreamC &s,PropositionSetC &obj) {
    obj = PropositionSetC(s);
    return s;
  }
 
  static TypeNameC type1(typeid(PropositionSetC),"RavlProbN::PropositionSetC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(PropositionSetBodyC,PropositionSetC,RCHandleVC<PropositionSetBodyC>);
  
}
