# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2007-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
PACKAGE = Ravl/Image

DONOT_SUPPORT=VCPP

REQUIRES = OpenCV

MAINS = exOpenCV.cc

HEADERS = OpenCVConvert.hh

SOURCES = OpenCVConvert.cc

PLIB = RavlOpenCV

EXTERNALLIBS = OpenCV.def

USESLIBS = RavlImage OpenCV 

PROGLIBS = RavlImageIO RavlDPDisplay.opt OpenCV

TESTEXES = testOpenCV.cc

TESTLIBS = OpenCV

EXAMPLES = exOpenCV.cc

EHT = OpenCV.html
