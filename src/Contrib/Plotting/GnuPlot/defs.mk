# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

DONOT_SUPPORT=VCPP

PACKAGE=Ravl/Plot

MAINS = exGnuPlot.cc exGnuPlot3d.cc

HEADERS = GnuPlot.hh GnuPlot3d.hh

REQUIRES = GnuPlot

SOURCES = GnuPlot.cc GnuPlot3d.cc

PLIB = RavlPlot

USESLIBS=  RavlCore RavlMath RavlOS RavlIO RavlRLog

PROGLIBS= RavlExtImgIO RavlDPDisplay
