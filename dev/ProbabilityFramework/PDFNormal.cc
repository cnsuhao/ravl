// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/PDFNormal.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PDFNormalBodyC::PDFNormalBodyC(const RandomVariableContinuousC& variable, RealT mean, RealT variance)
    : PDFContinuousAbstractBodyC(variable)
  {
    SetMeanAndVariance(mean, variance);
  }

  PDFNormalBodyC::~PDFNormalBodyC() {
  }

  RealT PDFNormalBodyC::MeasureProbability(RealT value) const {
    return m_constant1 * Exp(Sqr(value - m_mean) * m_constant2);
  }

  void PDFNormalBodyC::SetMeanAndVariance(RealT mean, RealT variance) {
    m_mean = mean;
    m_constant1 = 1.0 / (Sqrt (variance * 2.0 * RavlConstN::pi));
    m_constant2 = -1.0 / (2.0 * variance);
  }

}
