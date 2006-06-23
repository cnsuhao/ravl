// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableDiscrete.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableDiscreteBodyC::RandomVariableDiscreteBodyC(const StringC& name, const HSetC<StringC>& values)
    : RandomVariableBodyC(name)
  {
    SetValues(values);
  }

  RandomVariableDiscreteBodyC::RandomVariableDiscreteBodyC(const StringC& name)
    : RandomVariableBodyC(name)
  {
  }

  RandomVariableDiscreteBodyC::RandomVariableDiscreteBodyC(istream &in)
    : RandomVariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableDiscreteBodyC(istream &), Unrecognised version number in stream.");
    HSetC<StringC> values;
    in >> values;
    SetValues(values);
  }

  RandomVariableDiscreteBodyC::RandomVariableDiscreteBodyC(BinIStreamC &in)
    : RandomVariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableDiscreteBodyC(BinIStream &), Unrecognised version number in stream.");
    HSetC<StringC> values;
    in >> values;
    SetValues(values);
  }
  
  bool RandomVariableDiscreteBodyC::Save (ostream &out) const {
    if(!RandomVariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Values();
    return true;
  }
  
  bool RandomVariableDiscreteBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Values();
    return true;
  }

  RandomVariableDiscreteBodyC::~RandomVariableDiscreteBodyC() {
  }
  
  StringC RandomVariableDiscreteBodyC::ToString() const {
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

  SizeT RandomVariableDiscreteBodyC::NumValues() const {
    return m_numValues;
  }

  const HSetC<StringC>& RandomVariableDiscreteBodyC::Values() const {
    return m_values;
  }

  const StringC& RandomVariableDiscreteBodyC::Value(IndexC index) const {
    if (index > Values().Size())
      throw ExceptionC("RandomVariableDiscreteBodyC::Value(), index too big");
    HSetIterC<StringC> it(Values());
    while(index--)
      it++;
    return *it;
  }

  IndexC RandomVariableDiscreteBodyC::Index(const StringC& value) const {
    //:FIXME- this should probably be implemented using hash table
    IndexC index(0);
    for (HSetIterC<StringC> it(Values()); it; it++, index++)
      if (*it == value)
        return index;
    throw ExceptionC("RandomVariableDiscreteBodyC::Index(), couldn't find value");
  }

  void RandomVariableDiscreteBodyC::SetValues(const HSetC<StringC>& values) {
    HSetC<StringC> downcaseValues;
    for (HSetIterC<StringC> it(values); it; it++)
      downcaseValues.Insert(downcase(*it));
    m_numValues = downcaseValues.Size();
    m_values = downcaseValues;
  }

  static TypeNameC type1(typeid(RandomVariableDiscreteC),"RavlProbN::RandomVariableDiscreteC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableDiscreteBodyC,RandomVariableDiscreteC,RandomVariableC);
  
}
