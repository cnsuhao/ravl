// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/BayesianNetworkAbstract.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/Prob/RandomVariableValueDiscrete.hh"
#include "Ravl/Prob/PDFDiscrete.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlProbN {
  using namespace RavlN;
  
  BayesianNetworkAbstractBodyC::BayesianNetworkAbstractBodyC(const RCHashC<RandomVariableC,ConditionalProbabilityDistributionC>& nodeCPDs) {
    RCHashC<RandomVariableC,ConditionalProbabilityDistributionC> nodeCPDsCopy = nodeCPDs.Copy();
    m_nodeCPDs = nodeCPDs.Copy();
    HSetC<RandomVariableC> variables;
    while (nodeCPDsCopy.Size() > 0) {
      UIntT sizeCPDs = nodeCPDsCopy.Size();
      for (HashIterC<RandomVariableC,ConditionalProbabilityDistributionC> ht(nodeCPDsCopy); ht; ht++) {
        // check if all parents of current node are already in variables
        bool parentsAlreadyPresent = true;
        for (HSetIterC<RandomVariableC> st(ht.Data().ParentDomain().Variables()); st; st++) {
          if (!variables.Contains(*st)) {
            parentsAlreadyPresent = false;
            break;
          }
        }
        if (parentsAlreadyPresent == true) {
          variables.Insert(ht.Key());
          m_orderedNodes.InsLast(ht.Key());
          ht.Del();
          break; // because the iterator is probably invalid
        }
      }
      if (nodeCPDsCopy.Size() == sizeCPDs)
        throw ExceptionC("BayesianNetworkAbstractBodyC::BayesianNetworkAbstractBodyC(), graph must be acyclic");
    }
#if DODEBUG
    for (DLIterC<RandomVariableC> it(m_orderedNodes); it; it++)
      SysLog(SYSLOG_DEBUG) << "BayesianNetworkAbstractBodyC::BayesianNetworkAbstractBodyC(), " << it->ToString();
#endif
    m_domain = DomainC(variables);
  }

  BayesianNetworkAbstractBodyC::~BayesianNetworkAbstractBodyC() {
  }

  //: This function's implementation is based on ENUMERATION-ASK(X,e,bn) from
  //: Figure 14.9 in Artificial Intelligence: A Modern Approach, 2nd edition

  ProbabilityDistributionC BayesianNetworkAbstractBodyC::CalculateDistribution(const RandomVariableC& variable, const PropositionC& evidence) const {
    RandomVariableDiscreteC discrete(variable);
    if (!discrete.IsValid())
      throw ExceptionC("BayesianNetworkSimpleBodyC::CalculateDistribution(), only works for discrete variables");
    // check if evidence contains variable
    RandomVariableValueDiscreteC prior;
    for (HSetIterC<RandomVariableValueC> ht(evidence.Values()); ht; ht++) {
      if (ht->RandomVariable() == discrete)
        prior = *ht;
    }
    // calculate probability of each value independently
    RealT sum = 0;
    RCHashC<RandomVariableValueDiscreteC,RealT> probabilityLookupTable;
    for (HSetIterC<StringC> ht(discrete.Values()); ht; ht++) {
      RandomVariableValueDiscreteC value(discrete, *ht);    
      RealT probability;
      if (!prior.IsValid()) {
        PropositionC allEvidence(evidence, value);
        probability = CalculateProbability(allEvidence);
      }
      else {
        probability = (value == prior);
      }
      sum += probability;
      probabilityLookupTable.Insert(value,probability);
    }
    // normalise the values to sum to 1.0
    RealT alpha = 1.0 / sum;
    for (HashIterC<RandomVariableValueDiscreteC,RealT> ht(probabilityLookupTable); ht; ht++) {
      ht.Data() *= alpha;
    }
    return PDFDiscreteC(discrete, probabilityLookupTable);
  }

  DomainC BayesianNetworkAbstractBodyC::Domain() const {
    return m_domain; 
  }

  DListC<RandomVariableC> BayesianNetworkAbstractBodyC::Variables(const PropositionC& evidence) const {
    //:FIXME- this ought to consider the markov blanket of the evidence
    return m_orderedNodes;
  }

  ConditionalProbabilityDistributionC BayesianNetworkAbstractBodyC::NodeCPD(const RandomVariableC& variable) const {
    return m_nodeCPDs[variable];
  }

}
