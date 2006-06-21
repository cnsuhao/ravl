// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/Proposition.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PropositionBodyC::PropositionBodyC(const DomainC& domain, const HSetC<RandomVariableValueC>& values) {
    SetDomain(domain);
    SetValues(values);
  }

  PropositionBodyC::PropositionBodyC(const PropositionBodyC& other, const RandomVariableValueC& value) {
    SetDomain(other.Domain());
    HSetC<RandomVariableValueC> values = other.Values().Copy();
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
    DomainC domain(in);
    SetDomain(domain);
    HSetC<RandomVariableValueC> values;
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
    DomainC domain(in);
    SetDomain(domain);
    HSetC<RandomVariableValueC> values;
    in >> values;
    SetValues(values);
  }
  
  bool PropositionBodyC::Save (ostream &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Domain() << ' ' << Values();
    return true;
  }
  
  bool PropositionBodyC::Save (BinOStreamC &out) const {
    if(!RCBodyVC::Save(out))
      return false;
    IntT version = 0;
    out << version << Domain() << Values();
    return true;
  }

  PropositionBodyC::~PropositionBodyC() {
  }

  const DomainC& PropositionBodyC::Domain() const {
    return m_domain;
  }

  SizeT PropositionBodyC::NumValues() const {
    return Values().Size();
  }

  const HSetC<RandomVariableValueC>& PropositionBodyC::Values() const {
    return m_values;
  }

  const RandomVariableValueC& PropositionBodyC::Value(IndexC index) const {
    if (index > NumValues())
      throw ExceptionC("PropositionBodyC::Value(), index too big");
    HSetIterC<RandomVariableValueC> it(Values());
    while(index--)
      it++;
    return *it;
  }

  StringC PropositionBodyC::ToString() const {
    HSetIterC<RandomVariableValueC> it(Values());
    StringC string = it->RandomVariable().Name() + "=" + it->ToString();
    for (it++; it; it++) {
      string += ",";
      string += it->RandomVariable().Name() + "=" + it->ToString();
    }
    return string;
  }

  StringC PropositionBodyC::LotteryName() const {
    StringC name = Domain().ToString(); // show all variables
    HSetC<RandomVariableC> domainSet = Domain().Variables().Copy();
    for (HSetIterC<RandomVariableValueC> it(Values()); it; it++) {
      domainSet.Remove(it->RandomVariable());
    }
    name += "->(";
    if (domainSet.Size() > 0) {
      HSetIterC<RandomVariableC> it(domainSet);
      name += it->Name();
      for (it++; it; it++) {
        name += ",";
        name += it->Name();
      }
    }
    name += ")";
    return name;
  }

  PropositionC PropositionBodyC::SubProposition(const DomainC& subDomain) const {
    HSetC<RandomVariableValueC> values;
    for (HSetIterC<RandomVariableValueC> ht(Values()); ht; ht++) {
      if (subDomain.Variables().Contains(ht->RandomVariable()))
        values.Insert(*ht);
    }
    return PropositionC(subDomain, values);
  }

  void PropositionBodyC::SetDomain(const DomainC& domain) {
    m_domain = domain;
  }

  void PropositionBodyC::SetValues(const HSetC<RandomVariableValueC>& values) {
    //:FIXME- what collection for efficiency?
    for (HSetIterC<RandomVariableValueC> it(values); it; it++)
      if (!Domain().Contains(it->RandomVariable()))
        throw ExceptionC("PropositionBodyC::SetValues(), value not in domain");
    m_values = values.Copy();
  }

  bool PropositionBodyC::operator==(const PropositionBodyC& other) const {
    if (this == &other)
      return true;
    if (Domain() != other.Domain())
      return false;
    if (NumValues() != other.NumValues())
      return false;
    for (HSetIterC<RandomVariableValueC> ht(Values()); ht; ht++)
      if (!other.Values().Contains(*ht))
        return false;
    return true;
    
  }

  UIntT PropositionBodyC::Hash() const {
    UIntT hash = 0;
    for (HSetIterC<RandomVariableValueC> it(Values()); it; it++)
      hash += it->Hash();
    return Domain().Hash() + hash;
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
