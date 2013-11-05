# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Image/Processing/Filters/defs.mk"

PACKAGE=Ravl/Image

MAINS=doHomomorphicFilter.cc unDistort.cc 

HEADERS=  HomomorphicFilter.hh HistogramEqualise.hh \
 PixelMixer.hh RemoveDistortion.hh DeinterlaceStream.hh \
 DCT2d.hh ImageExtend.hh ImagePyramid.hh MedianFilter.hh

SOURCES= HomomorphicFilter.cc  DCT2d.cc DeinterlaceStream.cc \
 MedianFilter.cc

EXAMPLES = exImagePyramid.cc

TESTEXES = testDeinterlace.cc

LIBDEPS=RavlImageMisc.def

PLIB=RavlImageProc

SUMMARY_LIB=Ravl

USESLIBS=RavlImageProc

PROGLIBS=RavlIO RavlImageIO RavlExtImgIO.opt RavlOSIO RavlOptimise  RavlThreads RavlDPDisplay.opt DynLink


