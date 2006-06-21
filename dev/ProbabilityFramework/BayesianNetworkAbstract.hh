// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_BAYESIANNETWORKABSTRACT_HEADER
#define RAVLPROB_BAYESIANNETWORKABSTRACT_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/BayesianNetwork.hh"
#include "Ravl/DList.hh"
#include "Omni/Prob/ConditionalProbabilityDistribution.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Abstract implementation for bayesian network pulling in internal network
  //: representation
  class BayesianNetworkAbstractBodyC
    : public BayesianNetworkBodyC {
  public:
    BayesianNetworkAbstractBodyC(const RCHashC<RandomVariableC,ConditionalProbabilityDistributionC>& nodeCPDs);
    //: Constructor

    virtual ~BayesianNetworkAbstractBodyC();
    //: Destructor
    
    virtual ProbabilityDistributionC CalculateDistribution(const RandomVariableC& variable, const PropositionC& evidence) const;
    //: Calculate the a pdf to represent P(Variable|evidence)
    //!param: variable - a random variable that we want the distribution for
    //!param: evidence - a proposition with fixed values for some (or all) evidence variables from the network
    //!return: the probability distribution of Variable given the evidence

    virtual DomainC Domain() const;
    //: Get the domain for the network

  protected:
    DListC<RandomVariableC> Variables(const PropositionC& evidence) const;
    ConditionalProbabilityDistributionC NodeCPD(const RandomVariableC& variable) const;

  private:
    DomainC m_domain;
    //: The domain for the network

    DListC<RandomVariableC> m_orderedNodes;
    //: Ordered list of random variables in the network domain

    RCHashC<RandomVariableC,ConditionalProbabilityDistributionC> m_nodeCPDs;
    //: Map of variables to their conditional probability distributions
  };

  //! userlevel=Normal
  //: Abstract implementation for bayesian network pulling in internal network
  //: representation
  //!cwiz:author
  
  class BayesianNetworkAbstractC
    : public BayesianNetworkC
  {
  public:
    BayesianNetworkAbstractC()
    {}
    //: Default constructor makes invalid handle

  protected:
    BayesianNetworkAbstractC(BayesianNetworkAbstractBodyC &bod)
     : BayesianNetworkC(bod)
    {}
    //: Body constructor. 
    
    BayesianNetworkAbstractC(const BayesianNetworkAbstractBodyC *bod)
     : BayesianNetworkC(bod)
    {}
    //: Body constructor. 
    
    BayesianNetworkAbstractBodyC& Body()
    { return static_cast<BayesianNetworkAbstractBodyC &>(BayesianNetworkC::Body()); }
    //: Body Access. 
    
    const BayesianNetworkAbstractBodyC& Body() const
    { return static_cast<const BayesianNetworkAbstractBodyC &>(BayesianNetworkC::Body()); }
    //: Body Access. 
    
  };

}

#endif
