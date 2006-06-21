// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/RandomVariableContinuous.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  RandomVariableContinuousBodyC::RandomVariableContinuousBodyC(const StringC& name, const RealRangeC& interval)
    : RandomVariableBodyC(name)
  {
    SetInterval(interval);
  }

  RandomVariableContinuousBodyC::RandomVariableContinuousBodyC(istream &in)
    : RandomVariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableContinuousBodyC(istream &), Unrecognised version number in stream.");
    RealRangeC interval;
    in >> interval;
    SetInterval(interval);
  }

  RandomVariableContinuousBodyC::RandomVariableContinuousBodyC(BinIStreamC &in)
    : RandomVariableBodyC(in)
  {
    IntT version;
    in >> version;
    if (version < 0 || version > 0)
      throw ExceptionOutOfRangeC("RandomVariableContinuousBodyC(BinIStream &), Unrecognised version number in stream.");
    RealRangeC interval;
    in >> interval;
    SetInterval(interval);
  }
  
  bool RandomVariableContinuousBodyC::Save (ostream &out) const {
    if(!RandomVariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << ' ' << version << ' ' << Interval();
    return true;
  }
  
  bool RandomVariableContinuousBodyC::Save (BinOStreamC &out) const {
    if(!RandomVariableBodyC::Save(out))
      return false;
    IntT version = 0;
    out << version << Interval();
    return true;
  }

  RandomVariableContinuousBodyC::~RandomVariableContinuousBodyC() {
  }
  
  StringC RandomVariableContinuousBodyC::ToString() const {
    StringC values = Name() + "=[";
    values += StringC(m_interval.Min());
    values += ",";
    values += StringC(m_interval.Max());
    values += "]";
    return values;
  }

  const RealRangeC& RandomVariableContinuousBodyC::Interval() const {
    return m_interval;
  }

  void RandomVariableContinuousBodyC::SetInterval(const RealRangeC& interval) {
    m_interval = interval;
  }

  static TypeNameC type1(typeid(RandomVariableContinuousC),"RavlProbN::RandomVariableContinuousC");
    
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(RandomVariableContinuousBodyC,RandomVariableContinuousC,RandomVariableC);
  
}
