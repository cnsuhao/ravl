// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/PDFUniform.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PDFUniformBodyC::PDFUniformBodyC(const RandomVariableContinuousC& variable, const RealRangeC& interval)
    : PDFContinuousAbstractBodyC(variable)
  {
    SetInterval(RandomVariableContinuous().Interval());
  }

  PDFUniformBodyC::~PDFUniformBodyC() {
  }

  RealT PDFUniformBodyC::MeasureProbability(RealT value) const {
    return m_interval.Contains(value)? m_probability: 0.0;
  }

  void PDFUniformBodyC::SetInterval(const RealRangeC& interval) {
    m_probability = 1.0 / interval.Size();
    m_interval = interval;
  }

}
