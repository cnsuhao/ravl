# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"
#! author="Robert Crida"

LICENSE=LGPL

ORGANISATION=SKA/KAT.

DESCRIPTION=Pattern recognition software

DONOT_SUPPORT=arm mingw VCPP

PACKAGE=Ravl/Prob

HEADERS= CPDAbstract.hh \
         CPDContinuousDiscrete1.hh \
         CPDDiscreteDiscrete.hh \
         CPDDesigner.hh \
         CPDDesignerContinuousDiscrete1.hh \
         CPDDesignerFactory.hh \
         CPDPriorPDF.hh \
         ConditionalProbabilityDistribution.hh \
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
         ProbabilityDistribution.hh

SOURCES= CPDAbstract.cc \
         CPDContinuousDiscrete1.cc \
         CPDDiscreteDiscrete.cc \
         CPDDesigner.cc \
         CPDDesignerContinuousDiscrete1.cc \
         CPDDesignerFactory.cc \
         CPDPriorPDF.cc \
         ConditionalProbabilityDistribution.cc \
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
         ProbabilityDistribution.cc

MAINS=

USESLIBS=RavlCore RavlOS

TESTEXES=

PROGLIBS=RavlExtImgIO.opt RavlDPDisplay CPPUnit

PLIB=RavlProb
