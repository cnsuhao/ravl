# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

DONOT_SUPPORT=VCPP

PACKAGE=Ravl/Plot

MAINS = exGnuPlot.cc exGnuPlot2d.cc exGnuPlot3d.cc exGnuPlot2dClassifier.cc

HEADERS = GnuPlot2d.hh GnuPlot.hh GnuPlot3d.hh

REQUIRES =

SOURCES = GnuPlot2d.cc GnuPlot.cc GnuPlot3d.cc

PLIB = RavlGnuPlot

USESLIBS=  RavlPlot RavlOS

PROGLIBS= RavlExtImgIO RavlDPDisplay RavlPatternRec
