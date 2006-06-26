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
  
  CPDAbstractBodyC::CPDAbstractBodyC(const VariableC& variable, const DomainC& parentDomain) {
    SetVariable(variable);
    SetParentDomain(parentDomain);
  }

  CPDAbstractBodyC::CPDAbstractBodyC(const VariableC& variable, const VariableC& parentVariable) {
    SetVariable(variable);
    SetSingleParentVariable(parentVariable);
  }

  CPDAbstractBodyC::~CPDAbstractBodyC() {
  }

  RealT CPDAbstractBodyC::ConditionalProbability(const RandomVariableValueC& value, const PropositionC& parentValues) const {
    if (value.Variable() != Variable())
      throw ExceptionC("CPDAbstractBodyC::ConditionalProbability(), value isn't of correct variable");
    ProbabilityDistributionC pdf(ConditionalDistribution(parentValues));
    return pdf.MeasureProbability(value);
  }

  VariableC CPDAbstractBodyC::Variable() const {
    return m_variable;
  }

  DomainC CPDAbstractBodyC::ParentDomain() const {
    return m_parentDomain;
  }

  void CPDAbstractBodyC::SetVariable(const VariableC& variable) {
    m_variable = variable;
  }

  void CPDAbstractBodyC::SetParentDomain(const DomainC& parentDomain) {
    m_parentDomain = parentDomain;
  }

  void CPDAbstractBodyC::SetSingleParentVariable(const VariableC& parentVariable) {
    HSetC<VariableC> parents;
    parents.Insert(parentVariable);
    m_parentDomain = DomainC(parents);
  }

}
