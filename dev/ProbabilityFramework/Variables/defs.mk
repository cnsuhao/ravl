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

PACKAGE=Ravl/Prob

HEADERS= Variable.hh \
         VariableBoolean.hh \
         RandomVariableContinuous.hh \
         VariableDiscrete.hh \
         RandomVariableValue.hh \
         RandomVariableValueBoolean.hh \
         RandomVariableValueContinuous.hh \
         RandomVariableValueDiscrete.hh

SOURCES= Variable.cc \
         VariableBoolean.cc \
         RandomVariableContinuous.cc \
         VariableDiscrete.cc \
         RandomVariableValue.cc \
         RandomVariableValueBoolean.cc \
         RandomVariableValueContinuous.cc \
         RandomVariableValueDiscrete.cc

MAINS=

USESLIBS=RavlCore RavlOS

TESTEXES=

PROGLIBS=RavlExtImgIO.opt RavlDPDisplay

PLIB=RavlProb
