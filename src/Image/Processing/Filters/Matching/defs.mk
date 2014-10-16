# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE=Ravl/Image

HEADERS= Correlate2d.hh NormalisedCorrelation.hh Matching.hh 

SOURCES= Correlate2d.cc Matching.cc 

LIBDEPS=RavlImageMatching.def

PLIB=RavlImageProc

USESLIBS=RavlImageProc

