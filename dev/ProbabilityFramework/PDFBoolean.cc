// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/PDFBoolean.hh"
#include "Omni/Prob/RandomVariableValueBoolean.hh"

namespace RavlProbN {
  using namespace RavlN;
  
  PDFBooleanBodyC::PDFBooleanBodyC(const RandomVariableBooleanC& variable, RealT probabilityTrue)
    : PDFDiscreteBodyC(variable)
  {
    RCHashC<RandomVariableValueDiscreteC,RealT> probabilityLookupTable;
    probabilityLookupTable.Insert(RandomVariableValueBooleanC(variable, true), probabilityTrue);
    probabilityLookupTable.Insert(RandomVariableValueBooleanC(variable, false), 1.0-probabilityTrue);
    SetProbabilityLookupTable(probabilityLookupTable);
  }

  PDFBooleanBodyC::~PDFBooleanBodyC() {
  }

}
