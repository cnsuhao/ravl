// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#include "Ravl/Prob/BayesianNetworkSimple.hh"
#include "Ravl/Prob/RandomVariableValueBoolean.hh"
#include "Ravl/Prob/PDFDiscrete.hh"
#include "Ravl/Prob/PDFBoolean.hh"
#include "Ravl/Prob/CPDPriorPDF.hh"
#include "Ravl/Prob/CPDDiscreteDiscrete.hh"
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

using namespace RavlProbN;

int main() {
  try {
    // create nodes
    RandomVariableBooleanC Burglary("Burglary");
    RandomVariableBooleanC Earthquake("Earthquake");
    RandomVariableBooleanC Alarm("Alarm");
    RandomVariableBooleanC JohnCalls("JohnCalls");
    RandomVariableBooleanC MaryCalls("MaryCalls");
    // create parent domains
    HSetC<RandomVariableC> alarmParents;
    alarmParents.Insert(Burglary);
    alarmParents.Insert(Earthquake);
    DomainC alarmParentDomain(alarmParents);
    HSetC<RandomVariableC> johnCallsParents;
    johnCallsParents.Insert(Alarm);
    DomainC johnCallsParentDomain(johnCallsParents);
    DomainC maryCallsParentDomain(johnCallsParents);
    // create variable values
    RandomVariableValueBooleanC burglary(Burglary, true);
    RandomVariableValueBooleanC _burglary(Burglary, false);
    RandomVariableValueBooleanC earthquake(Earthquake, true);
    RandomVariableValueBooleanC _earthquake(Earthquake, false);
    RandomVariableValueBooleanC alarm(Alarm, true);
    RandomVariableValueBooleanC _alarm(Alarm, false);
    RandomVariableValueBooleanC johnCalls(JohnCalls, true);
    RandomVariableValueBooleanC _johnCalls(JohnCalls, false);
    RandomVariableValueBooleanC maryCalls(MaryCalls, true);
    RandomVariableValueBooleanC _maryCalls(MaryCalls, false);
    // create distribution tables
    HSetC<RandomVariableValueC> be; be.Insert(burglary); be.Insert(earthquake);
    HSetC<RandomVariableValueC> b_e; b_e.Insert(burglary); b_e.Insert(_earthquake);
    HSetC<RandomVariableValueC> _be; _be.Insert(_burglary); _be.Insert(earthquake);
    HSetC<RandomVariableValueC> _b_e; _b_e.Insert(_burglary); _b_e.Insert(_earthquake);
    HSetC<RandomVariableValueC> a; a.Insert(alarm);
    HSetC<RandomVariableValueC> _a; _a.Insert(_alarm);
    RCHashC<PropositionC,PDFDiscreteC> alarmPDFs;
    alarmPDFs.Insert(PropositionC(alarmParentDomain, be), PDFBooleanC(Alarm, 0.95));
    alarmPDFs.Insert(PropositionC(alarmParentDomain, b_e), PDFBooleanC(Alarm, 0.94));
    alarmPDFs.Insert(PropositionC(alarmParentDomain, _be), PDFBooleanC(Alarm, 0.29));
    alarmPDFs.Insert(PropositionC(alarmParentDomain, _b_e), PDFBooleanC(Alarm, 0.001));
    RCHashC<PropositionC,PDFDiscreteC> johnCallsPDFs;
    johnCallsPDFs.Insert(PropositionC(johnCallsParentDomain, a), PDFBooleanC(JohnCalls, 0.90));
    johnCallsPDFs.Insert(PropositionC(johnCallsParentDomain, _a), PDFBooleanC(JohnCalls, 0.05));
    RCHashC<PropositionC,PDFDiscreteC> maryCallsPDFs;
    maryCallsPDFs.Insert(PropositionC(maryCallsParentDomain, a), PDFBooleanC(MaryCalls, 0.70));
    maryCallsPDFs.Insert(PropositionC(maryCallsParentDomain, _a), PDFBooleanC(MaryCalls, 0.01));
    // create CPDs
    CPDPriorPDFC burglaryCPT(PDFBooleanC(Burglary, 0.001));
    CPDPriorPDFC earthquakeCPT(PDFBooleanC(Earthquake, 0.002));
    CPDDiscreteDiscreteC alarmCPT(Alarm, alarmParentDomain, alarmPDFs);
    CPDDiscreteDiscreteC johnCallsCPT(JohnCalls, johnCallsParentDomain, johnCallsPDFs);
    CPDDiscreteDiscreteC maryCallsCPT(MaryCalls, maryCallsParentDomain, maryCallsPDFs);
    // create the bayesian network
    RCHashC<RandomVariableC,ConditionalProbabilityDistributionC> nodeCPDs;
    nodeCPDs.Insert(Burglary, burglaryCPT);
    nodeCPDs.Insert(Earthquake, earthquakeCPT);
    nodeCPDs.Insert(Alarm, alarmCPT);
    nodeCPDs.Insert(JohnCalls, johnCallsCPT);
    nodeCPDs.Insert(MaryCalls, maryCallsCPT);
    BayesianNetworkSimpleC bayesianNetwork(nodeCPDs);

    // evaluate P(B|johnCalls,maryCalls)
    HSetC<RandomVariableValueC> propSet;
    propSet.Insert(johnCalls);
    propSet.Insert(maryCalls);
    PropositionC prop(bayesianNetwork.Domain(), propSet);
    ProbabilityDistributionC pdf = bayesianNetwork.CalculateDistribution(Burglary, PropositionC(bayesianNetwork.Domain(), propSet));

    cerr << pdf.ToString() << endl;
  }
  catch(ExceptionC& e) {
    cerr << "Caught exception: " << e.Text() << endl;
  }
}
