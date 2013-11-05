# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Image/Processing/Lines/defs.mk"

PACKAGE = Ravl/Image

MAINS = doPPHT.cc doArcs.cc
#genSobelImg.cc

HEADERS= PixelMap.hh PixelMapSearch.hh PCPixel.hh PCPixelList.hh \
 PCMapping.hh PPHT.hh ArcDetector.hh WhiteLineDetector.hh

SOURCES= PixelMap.cc PixelMapSearch.cc PCPixel.cc PCPixelList.cc \
 PCMapping.cc PPHT.cc ArcDetector.cc WhiteLineDetector.cc

PLIB=RavlImageProc

SUMMARY_LIB=Ravl

LIBDEPS=RavlImageProcHT.def

USESLIBS=RavlImageProc 

PROGLIBS=RavlIO RavlOS RavlOSIO RavlImageIO RavlMathIO RavlImgIOV4L.opt RavlDPDisplay.opt RavlExtImgIO.opt DynLink

EHT = Ravl.API.Images.Lines.html 

EXAMPLES=doPPHT.cc doArcs.cc 

TESTEXES=testLineDet.cc
#exedoPPHT.eht
