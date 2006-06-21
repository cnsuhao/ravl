// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/CPDDiscreteDiscrete.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  CPDDiscreteDiscreteBodyC::CPDDiscreteDiscreteBodyC(const RandomVariableDiscreteC& randomVariable,
                                                     const DomainC& parentDomain,
                                                     const RCHashC<PropositionC,PDFDiscreteC>& probabilityDistributionTable)
    : CPDAbstractBodyC(randomVariable, parentDomain) {
    SetProbabilityDistributionTable(probabilityDistributionTable);
  }

  CPDDiscreteDiscreteBodyC::~CPDDiscreteDiscreteBodyC() {
  }

  ProbabilityDistributionC CPDDiscreteDiscreteBodyC::ConditionalDistribution(const PropositionC& parentValues) const {
    PDFDiscreteC pdf;
    HSetC<RandomVariableValueC> values = parentValues.Values();
    if (values.Size() != ParentDomain().NumVariables())
      throw ExceptionC("CPDDiscreteDiscreteBodyC::ConditionalDistribution(), called with incorrect proposition!");
    if (!m_probabilityDistributionTable.Lookup(parentValues, pdf))
      throw ExceptionC(StringC("CPDDiscreteDiscreteBodyC::ConditionalDistribution(), couldn't find distribution") + parentValues.ToString());
    return pdf;
  }
  
  void CPDDiscreteDiscreteBodyC::SetProbabilityDistributionTable(const RCHashC<PropositionC,PDFDiscreteC>& probabilityDistributionTable) {
    // ensure all parents are discrete and calculate combinations
    UIntT numCombinations = 1;
    for (HSetIterC<RandomVariableC> ht(ParentDomain().Variables()); ht; ht++) {
      RandomVariableDiscreteC discrete(*ht);
      if (!discrete.IsValid())
        throw ExceptionC("CPDDiscreteDiscreteBodyC::SetProbabilityDistributionTable(), all parents must be discrete");
      numCombinations *= discrete.NumValues();
    }
    // check that there is a table for each value
    if (numCombinations != probabilityDistributionTable.Size())
      throw ExceptionC("CPDDiscreteDiscreteBodyC::SetProbabilityDistributionTable(), need table for each combination of parents");
    // check that all tables are for the correct parents
    for (HashIterC<PropositionC,PDFDiscreteC> ht(probabilityDistributionTable); ht; ht++) {
      if (ht.Key().Domain() != ParentDomain())
        throw ExceptionC("CPDDiscreteDiscreteBodyC::SetProbabilityDistributionTable(), each table must be for parent domain");
      if (ht.Key().NumValues() != ParentDomain().NumVariables())
        throw ExceptionC("CPDDiscreteDiscreteBodyC::SetProbabilityDistributionTable(), each table must be for a complete combination of parent variables");
    }
    m_probabilityDistributionTable = probabilityDistributionTable.Copy();
  }

}
