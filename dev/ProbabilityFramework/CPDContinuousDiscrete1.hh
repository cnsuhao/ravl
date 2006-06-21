// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_CPDCONTINUOUSDISCRETE1_HEADER
#define RAVLPROB_CPDCONTINUOUSDISCRETE1_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/CPDAbstract.hh"
#include "Ravl/Prob/RandomVariableContinuous.hh"
#include "Ravl/Prob/RandomVariableValueDiscrete.hh"
#include "Ravl/Prob/PDFContinuousAbstract.hh"
#include "Ravl/RCHash.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Conditional probability distribution of a continuous variable with only a single discrete parent
  class CPDContinuousDiscrete1BodyC
    : public CPDAbstractBodyC {
  public:
    CPDContinuousDiscrete1BodyC(const RandomVariableContinuousC& randomVariable,
                                const RandomVariableDiscreteC& parentVariable,
                                const RCHashC<RandomVariableValueDiscreteC,PDFContinuousAbstractC>& probabilityDistributionTable);
    //: Constructor

    virtual ~CPDContinuousDiscrete1BodyC();
    //: Destructor
    
    virtual ProbabilityDistributionC ConditionalDistribution(const PropositionC& parentValues) const;
    //: Calculate the probability distribution for P(V|parentValues)
    //!param: parentValues - a proposition with fixed values for some (or all) parent variables
    //!return: the probability distribution of the random variable

  private:
    void SetProbabilityDistributionTable(const RCHashC<RandomVariableValueDiscreteC,PDFContinuousAbstractC>& probabilitDistributionTable);
    //: Set the table of distributions

    RandomVariableDiscreteC ParentVariable() const;
    //: Get access to the parent variable directly

    void SetParentVariable(const RandomVariableDiscreteC& parentVariable);
    //: Cache the parent variable for direct access

  private:
    RCHashC<RandomVariableValueDiscreteC,PDFContinuousAbstractC> m_probabilityDistributionTable;
    //: Table of distributions for each value of the discrete parent

    RandomVariableDiscreteC m_parentVariable;
    //: Cached copy of parent variable for direct access
  };

  //! userlevel=Normal
  //: Conditional probability distribution of a continuous variable with only a single discrete parent
  //!cwiz:author
  
  class CPDContinuousDiscrete1C
    : public CPDAbstractC
  {
  public:
    CPDContinuousDiscrete1C(const RandomVariableContinuousC& randomVariable,
                            const RandomVariableDiscreteC& parentVariable,
                            const RCHashC<RandomVariableValueDiscreteC,PDFContinuousAbstractC>& probabilityDistributionTable)
      : CPDAbstractC(new CPDContinuousDiscrete1BodyC(randomVariable, parentVariable, probabilityDistributionTable))
    {}
    //: Constructor

  };

}

#endif
