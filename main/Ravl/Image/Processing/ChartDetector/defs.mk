# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2003, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"

PACKAGE=Ravl/Image

MAINS=chartdet.cc

HEADERS= ChartDetector.hh ChartDetectorRegion.hh

SOURCES= ChartDetector.cc ChartDetectorRegion.cc

PLIB=RavlChartDetector

USESLIBS=RavlImage RavlImageProc RavlImageIO 

PROGLIBS=RavlDPDisplay
