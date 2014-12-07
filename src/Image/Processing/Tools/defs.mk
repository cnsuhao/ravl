# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE = Ravl/Image

MAINS = convertFeatureSet.cc

HEADERS = Rectangle2dIter.hh SobolImage.hh SubSample.hh SummedAreaTable.hh \
 SummedAreaTable2.hh PyramidScan.hh PeakDetector.hh \
 ImagePointFeatureSet.hh ImagePointFeature.hh TakeSubImage.hh

SOURCES = Rectangle2dIter.cc SobolImage.cc PyramidScan.cc \
ImagePointFeatureSet.cc ImagePointFeature.cc 

MUSTLINK = ImagePointFeatureIO.cc

PLIB = RavlImageProc

SUMMARY_LIB=Ravl

USESLIBS = RavlImage RavlThreads RavlMath RavlIO RavlOS RavlOSIO RavlOptimise \
 RavlCore 
# RavlOS is needed for DeadLineTimerC in PPHT code.

EXAMPLES = exFeatureSet.cc 

EHT = Ravl.API.Images.Misc.html Ravl.API.Images.Misc.XMLFormat.eht

TESTEXES = testImageTools.cc
