// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/PDFContinuousAbstract.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PDFContinuousAbstractBodyC::PDFContinuousAbstractBodyC(const VariableContinuousC& variable)
    : PDFAbstractBodyC(variable)
  {
  }

  PDFContinuousAbstractBodyC::~PDFContinuousAbstractBodyC() {
  }

  RealT PDFContinuousAbstractBodyC::MeasureProbability(const VariablePropositionC& value) const {
    if (value.Variable() != Variable())
      throw ExceptionC("ProbabilityDistributionContinuousBodyC::MeasureProbability(), value doesn't match variable of this distribution");
    return MeasureProbability(RandomVariableValueContinuousC(value));
  }

  RealT PDFContinuousAbstractBodyC::MeasureProbability(const RandomVariableValueContinuousC& value) const {
    if (!value.IsValid())
      throw ExceptionC("ProbabilityDistributionContinuousBodyC::MeasureProbability(), value object is not valid");
    return MeasureProbability(value.Value());
  }

  VariableContinuousC PDFContinuousAbstractBodyC::VariableContinuous() const {
    return (VariableContinuousC)Variable();
  }

  StringC PDFContinuousAbstractBodyC::ToString() const {
    RavlAssertMsg(0, "PDFContinuousAbstractBodyC::ToString(), not implemented");
    return "";
  }

}
