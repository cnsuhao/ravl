// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/VariableDiscrete.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  VariableDiscreteBodyC::VariableDiscreteBodyC(const StringC& name, const HSetC<StringC>& values)
    : VariableBodyC(name)
  {
    SetValues(values);
  }

  VariableDiscreteBodyC::VariableDiscreteBodyC(const StringC& name)
    : VariableBodyC(name)
  {
  }

  VariableDiscreteBodyC::VariableDiscreteBodyC(istream &in)
    : VariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("VariableDiscreteBodyC(istream &), Unrecognised version number in stream.");
    HSetC<StringC> values;
    in >> values;
    SetValues(values);
  }

  VariableDiscreteBodyC::VariableDiscreteBodyC(BinIStreamC &in)
    : VariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("VariableDiscreteBodyC(BinIStream &), Unrecognised version number in stream.");
    HSetC<StringC> values;
    in >> values;
    SetValues(values);
  }
  
  bool VariableDiscreteBodyC::Save (ostream &out) const {
    if(!VariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Values();
    return true;
  }
  
  bool VariableDiscreteBodyC::Save (BinOStreamC &out) const {
    if(!VariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Values();
    return true;
  }

  VariableDiscreteBodyC::~VariableDiscreteBodyC() {
  }
  
  StringC VariableDiscreteBodyC::ToString() const {
    StringC values = Name() + "=<";
    HSetIterC<StringC> it(Values());
    values += *it;
    for (it++ ; it; it++) {
      values += ",";
      values += *it;
    }
    values += ">";
    return values;
  }

  SizeT VariableDiscreteBodyC::NumValues() const {
    return m_numValues;
  }

  const HSetC<StringC>& VariableDiscreteBodyC::Values() const {
    return m_values;
  }

  const StringC& VariableDiscreteBodyC::Value(IndexC index) const {
    if (index < 0 || index >= Values().Size())
      throw ExceptionC("VariableDiscreteBodyC::Value(), index out of bounds");
    HSetIterC<StringC> it(Values());
    while(index--)
      it++;
    return *it;
  }

  IndexC VariableDiscreteBodyC::Index(const StringC& value) const {
    //:FIXME- this should probably be implemented using hash table
    IndexC index(0);
    for (HSetIterC<StringC> it(Values()); it; it++, index++)
      if (*it == value)
        return index;
    throw ExceptionC("VariableDiscreteBodyC::Index(), couldn't find value");
  }

  void VariableDiscreteBodyC::SetValues(const HSetC<StringC>& values) {
    HSetC<StringC> downcaseValues;
    for (HSetIterC<StringC> it(values); it; it++)
      downcaseValues.Insert(downcase(*it));
    m_numValues = downcaseValues.Size();
    m_values = downcaseValues;
  }

  static TypeNameC type1(typeid(VariableDiscreteC),"RavlProbN::VariableDiscreteC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(VariableDiscreteBodyC,VariableDiscreteC,VariableC);
  
}
