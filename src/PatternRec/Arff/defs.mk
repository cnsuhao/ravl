# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2004, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE = Ravl/PatternRec

MAINS    = cutArff.cc \
           glueArff.cc 

SOURCES  = ArffIO.cc 

HEADERS  = ArffIO.hh 

PLIB     = ArffIO

SUMMARY_LIB=Ravl

USESLIBS = RavlCore 

PROGLIBS = RavlOS
