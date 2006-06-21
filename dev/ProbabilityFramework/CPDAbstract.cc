// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/CPDAbstract.hh"
#include "Ravl/Prob/ProbabilityDistribution.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  CPDAbstractBodyC::CPDAbstractBodyC(const RandomVariableC& randomVariable, const DomainC& parentDomain) {
    SetRandomVariable(randomVariable);
    SetParentDomain(parentDomain);
  }

  CPDAbstractBodyC::CPDAbstractBodyC(const RandomVariableC& randomVariable, const RandomVariableC& parentVariable) {
    SetRandomVariable(randomVariable);
    SetSingleParentVariable(parentVariable);
  }

  CPDAbstractBodyC::~CPDAbstractBodyC() {
  }

  RealT CPDAbstractBodyC::ConditionalProbability(const RandomVariableValueC& value, const PropositionC& parentValues) const {
    if (value.RandomVariable() != RandomVariable())
      throw ExceptionC("CPDAbstractBodyC::ConditionalProbability(), value isn't of correct variable");
    ProbabilityDistributionC pdf(ConditionalDistribution(parentValues));
    return pdf.MeasureProbability(value);
  }

  RandomVariableC CPDAbstractBodyC::RandomVariable() const {
    return m_randomVariable;
  }

  DomainC CPDAbstractBodyC::ParentDomain() const {
    return m_parentDomain;
  }

  void CPDAbstractBodyC::SetRandomVariable(const RandomVariableC& randomVariable) {
    m_randomVariable = randomVariable;
  }

  void CPDAbstractBodyC::SetParentDomain(const DomainC& parentDomain) {
    m_parentDomain = parentDomain;
  }

  void CPDAbstractBodyC::SetSingleParentVariable(const RandomVariableC& parentVariable) {
    HSetC<RandomVariableC> parents;
    parents.Insert(parentVariable);
    m_parentDomain = DomainC(parents);
  }

}
