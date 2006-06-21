# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"
#! author="Robert Crida"

LICENSE=LGPL

ORGANISATION=OmniPerception Ltd.

DESCRIPTION=Pattern recognition software

DONOT_SUPPORT=arm mingw VCPP

PACKAGE=Omni/Prob

HEADERS= BayesianNetwork.hh \
         BayesianNetworkAbstract.hh \
         BayesianNetworkSimple.hh \
         CPDAbstract.hh \
         CPDContinuousDiscrete1.hh \
         CPDDiscreteDiscrete.hh \
         CPDDesigner.hh \
         CPDDesignerContinuousDiscrete1.hh \
         CPDDesignerFactory.hh \
         CPDPriorPDF.hh \
         ConditionalProbabilityDistribution.hh \
         Domain.hh \
         Lottery.hh \
         PDFContinuousDesigner.hh \
         PDFContinuousDesignerNormal.hh \
         PDFContinuousDesignerUniform.hh \
         PDFDesignerFactory.hh \
         PDFAbstract.hh \
         PDFBoolean.hh \
         PDFContinuousAbstract.hh \
         PDFDiscrete.hh \
         PDFNormal.hh \
         PDFUniform.hh \
         ProbabilityDistribution.hh \
         Proposition.hh \
         RandomVariable.hh \
         RandomVariableBoolean.hh \
         RandomVariableContinuous.hh \
         RandomVariableDiscrete.hh \
         RandomVariableValue.hh \
         RandomVariableValueBoolean.hh \
         RandomVariableValueContinuous.hh \
         RandomVariableValueDiscrete.hh

SOURCES= BayesianNetwork.cc \
         BayesianNetworkAbstract.cc \
         BayesianNetworkSimple.cc \
         CPDAbstract.cc \
         CPDContinuousDiscrete1.cc \
         CPDDiscreteDiscrete.cc \
         CPDDesigner.cc \
         CPDDesignerContinuousDiscrete1.cc \
         CPDDesignerFactory.cc \
         CPDPriorPDF.cc \
         ConditionalProbabilityDistribution.cc \
         Domain.cc \
         Lottery.cc \
         PDFContinuousDesigner.cc \
         PDFContinuousDesignerNormal.cc \
         PDFContinuousDesignerUniform.cc \
         PDFDesignerFactory.cc \
         PDFAbstract.cc \
         PDFBoolean.cc \
         PDFContinuousAbstract.cc \
         PDFDiscrete.cc \
         PDFNormal.cc \
         PDFUniform.cc \
         ProbabilityDistribution.cc \
         Proposition.cc \
         RandomVariable.cc \
         RandomVariableBoolean.cc \
         RandomVariableContinuous.cc \
         RandomVariableDiscrete.cc \
         RandomVariableValue.cc \
         RandomVariableValueBoolean.cc \
         RandomVariableValueContinuous.cc \
         RandomVariableValueDiscrete.cc

MAINS= testRandomVariable.cc testBayesianNetwork.cc

USESLIBS=RavlCore RavlOS

TESTEXES=

PROGLIBS=RavlExtImgIO.opt RavlDPDisplay 

PLIB=RavlProb
