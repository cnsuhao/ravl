# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Image/Processing/Edges/defs.mk"

DESCRIPTION =Edge detection

PACKAGE=Ravl/Image

MAINS= doEdgeDet.cc

EXAMPLES=doEdgeDet.cc

HEADERS=EdgeSobel.hh EdgeDeriche.hh SqrComposition.hh \
 EdgeNonMaxSuppression.hh Edgel.hh EdgeLink.hh \
 EdgeDetector.hh EdgeDetect.hh Gradient.hh

SOURCES=EdgeSobel.cc EdgeDeriche.cc SqrComposition.cc \
 EdgeNonMaxSuppression.cc Edgel.cc EdgeLink.cc \
 EdgeDetector.cc EdgeDetect.cc EdgeIO.cc

MUSTLINK=RavlEdgeIO.cc

PLIB=RavlImageProc

SUMMARY_LIB=Ravl

LIBDEPS=RavlImageEdges.def

USESLIBS=RavlImageProc 

PROGLIBS= RavlImageIO RavlDPMT RavlOSIO RavlVideoIO RavlIO RavlImageIO RavlDPDisplay.opt RavlExtImgIO.opt \
 RavlImgIOV4L.opt  RavlDV.opt RavlLibFFmpeg.opt DynLink

EHT=Ravl.API.Images.Edges.html

TESTEXES= testEdges.cc
