// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#include "Ravl/Prob/RandomVariableBoolean.hh"
#include "Ravl/Prob/RandomVariableContinuous.hh"
#include "Ravl/Prob/Domain.hh"
#include "Ravl/Prob/RandomVariableValueBoolean.hh"
#include "Ravl/Prob/RandomVariableValueContinuous.hh"
#include "Ravl/Prob/Proposition.hh"
#include "Ravl/Prob/Lottery.hh"
#include "Ravl/Prob/CPDDesignerFactory.hh"
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

using namespace RavlProbN;

int main() {
  try {
    RandomVariableBooleanC boolean("Face");
    cerr << boolean.ToString() << " " << boolean << endl;
    HSetC<StringC> values;
    values.Insert("Sunny");
    values.Insert("Cloudy");
    values.Insert("Rainy");
    RandomVariableDiscreteC discrete("Weather", values);
    cerr << discrete.ToString() << " " << discrete << endl;
    RandomVariableContinuousC continuous("Fraction", RealRangeC(0,1));
    cerr << continuous.ToString() << " " << continuous << endl;

    HSetC<RandomVariableC> domainList;
    domainList.Insert(boolean);
    domainList.Insert(discrete);
    domainList.Insert(continuous);
    DomainC domain(domainList);
    cerr << domain.ToString() << " " << domain << endl;
  
    RandomVariableValueBooleanC booleanValue(boolean,true);
    cerr << booleanValue.ToString() << " " << booleanValue << endl;
    RandomVariableValueDiscreteC discreteValue(discrete,"sunny");
    cerr << discreteValue.ToString() << " " << discreteValue << endl;
    RandomVariableValueContinuousC continuousValue(continuous,0.7);
    cerr << continuousValue.ToString() << " " << continuousValue << endl;

    HSetC<RandomVariableValueC> propositionList;
//    propositionList.Insert(booleanValue);
    propositionList.Insert(discreteValue);
    propositionList.Insert(continuousValue);
    PropositionC proposition(domain, propositionList);
    cerr << proposition.ToString() << " " << proposition << endl;
    cerr << proposition.LotteryName() << endl;

    RCHashC<StringC,RealT> outcomes;
    outcomes.Insert("face",0.2);
    outcomes.Insert("¬face", 0.5);
    LotteryC lottery(proposition.LotteryName(), outcomes);
    cerr << lottery.ToString() << endl;

    HSetC<RandomVariableC> discretePairDomainSet;
    discretePairDomainSet.Insert(boolean);
    discretePairDomainSet.Insert(discrete);
    DomainC discreteDomain(discretePairDomainSet);
    cerr << discreteDomain.ToString() << endl;

    HSetC<RandomVariableC> parentVariables;
    parentVariables.Insert(boolean);
    DomainC parentDomain(parentVariables);
    CPDDesignerC designer = CPDDesignerFactoryC::GetInstance().GetCPDDesigner(continuous, parentDomain);
    DListC<Tuple2C<RandomVariableValueC,PropositionC> > propositionPairs;
    RealRangeC interval = continuous.Interval();
    for (int i = 0; i < 100; i++) {
      RealT random1 = Random1();
      RealT continuousValue = interval.Min() + random1 * interval.Size();
      bool state = continuousValue < interval.Center();
      RandomVariableValueC variable = RandomVariableValueContinuousC(continuous, continuousValue);
      HSetC<RandomVariableValueC> parentList;
      parentList.Insert(RandomVariableValueBooleanC(boolean, state));
      PropositionC parentProposition(parentDomain, parentList);
      Tuple2C<RandomVariableValueC,PropositionC> propositionPair(variable, parentProposition);
      propositionPairs.InsLast(propositionPair);
      cerr << variable.ToString() << " " << parentProposition.ToString() << endl;
    }
    ConditionalProbabilityDistributionC distribution = designer.CreateCPD(propositionPairs);
    for (DLIterC<Tuple2C<RandomVariableValueC,PropositionC> > it(propositionPairs); it; it++) {
      cerr << it->Data1().ToString() << " ";
      cerr << it->Data2().ToString() << " ";
      cerr << distribution.ConditionalProbability(it->Data1(), it->Data2()) << endl;
    }

  }
  catch(ExceptionC& e) {
    cerr << "Caught exception: " << e.Text() << endl;
  }
}
