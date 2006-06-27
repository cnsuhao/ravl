// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#include "Ravl/Prob/BayesianNetworkSimple.hh"
#include "Ravl/Prob/VariablePropositionBoolean.hh"
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
    VariableBooleanC Burglary("Burglary");
    VariableBooleanC Earthquake("Earthquake");
    VariableBooleanC Alarm("Alarm");
    VariableBooleanC JohnCalls("JohnCalls");
    VariableBooleanC MaryCalls("MaryCalls");
    // create parent domains
    HSetC<VariableC> alarmParents;
    alarmParents.Insert(Burglary);
    alarmParents.Insert(Earthquake);
    VariableSetC alarmParentVariableSet(alarmParents);
    HSetC<VariableC> johnCallsParents;
    johnCallsParents.Insert(Alarm);
    VariableSetC johnCallsParentVariableSet(johnCallsParents);
    VariableSetC maryCallsParentVariableSet(johnCallsParents);
    // create variable values
    VariablePropositionBooleanC burglary(Burglary, true);
    VariablePropositionBooleanC _burglary(Burglary, false);
    VariablePropositionBooleanC earthquake(Earthquake, true);
    VariablePropositionBooleanC _earthquake(Earthquake, false);
    VariablePropositionBooleanC alarm(Alarm, true);
    VariablePropositionBooleanC _alarm(Alarm, false);
    VariablePropositionBooleanC johnCalls(JohnCalls, true);
    VariablePropositionBooleanC _johnCalls(JohnCalls, false);
    VariablePropositionBooleanC maryCalls(MaryCalls, true);
    VariablePropositionBooleanC _maryCalls(MaryCalls, false);
    // create distribution tables
    HSetC<VariablePropositionC> be; be.Insert(burglary); be.Insert(earthquake);
    HSetC<VariablePropositionC> b_e; b_e.Insert(burglary); b_e.Insert(_earthquake);
    HSetC<VariablePropositionC> _be; _be.Insert(_burglary); _be.Insert(earthquake);
    HSetC<VariablePropositionC> _b_e; _b_e.Insert(_burglary); _b_e.Insert(_earthquake);
    HSetC<VariablePropositionC> a; a.Insert(alarm);
    HSetC<VariablePropositionC> _a; _a.Insert(_alarm);
    RCHashC<PropositionC,PDFDiscreteC> alarmPDFs;
    alarmPDFs.Insert(PropositionC(alarmParentVariableSet, be), PDFBooleanC(Alarm, 0.95));
    alarmPDFs.Insert(PropositionC(alarmParentVariableSet, b_e), PDFBooleanC(Alarm, 0.94));
    alarmPDFs.Insert(PropositionC(alarmParentVariableSet, _be), PDFBooleanC(Alarm, 0.29));
    alarmPDFs.Insert(PropositionC(alarmParentVariableSet, _b_e), PDFBooleanC(Alarm, 0.001));
    RCHashC<PropositionC,PDFDiscreteC> johnCallsPDFs;
    johnCallsPDFs.Insert(PropositionC(johnCallsParentVariableSet, a), PDFBooleanC(JohnCalls, 0.90));
    johnCallsPDFs.Insert(PropositionC(johnCallsParentVariableSet, _a), PDFBooleanC(JohnCalls, 0.05));
    RCHashC<PropositionC,PDFDiscreteC> maryCallsPDFs;
    maryCallsPDFs.Insert(PropositionC(maryCallsParentVariableSet, a), PDFBooleanC(MaryCalls, 0.70));
    maryCallsPDFs.Insert(PropositionC(maryCallsParentVariableSet, _a), PDFBooleanC(MaryCalls, 0.01));
    // create CPDs
    CPDPriorPDFC burglaryCPT(PDFBooleanC(Burglary, 0.001));
    CPDPriorPDFC earthquakeCPT(PDFBooleanC(Earthquake, 0.002));
    CPDDiscreteDiscreteC alarmCPT(Alarm, alarmParentVariableSet, alarmPDFs);
    CPDDiscreteDiscreteC johnCallsCPT(JohnCalls, johnCallsParentVariableSet, johnCallsPDFs);
    CPDDiscreteDiscreteC maryCallsCPT(MaryCalls, maryCallsParentVariableSet, maryCallsPDFs);
    // create the bayesian network
    RCHashC<VariableC,ConditionalProbabilityDistributionC> nodeCPDs;
    nodeCPDs.Insert(Burglary, burglaryCPT);
    nodeCPDs.Insert(Earthquake, earthquakeCPT);
    nodeCPDs.Insert(Alarm, alarmCPT);
    nodeCPDs.Insert(JohnCalls, johnCallsCPT);
    nodeCPDs.Insert(MaryCalls, maryCallsCPT);
    BayesianNetworkSimpleC bayesianNetwork(nodeCPDs);

    // evaluate P(B|johnCalls,maryCalls)
    HSetC<VariablePropositionC> propSet;
    propSet.Insert(johnCalls);
    propSet.Insert(maryCalls);
    PropositionC prop(bayesianNetwork.VariableSet(), propSet);
    ProbabilityDistributionC pdf = bayesianNetwork.CalculateDistribution(Burglary, PropositionC(bayesianNetwork.VariableSet(), propSet));

    cerr << pdf.ToString() << endl;
  }
  catch(ExceptionC& e) {
    cerr << "Caught exception: " << e.Text() << endl;
  }
}
