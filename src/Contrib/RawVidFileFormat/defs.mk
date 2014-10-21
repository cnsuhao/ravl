# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE = Ravl/Image

DONOT_SUPPORT=VCPP

DESCRIPTION = Raw video IO classes.

HEADERS = ImgIOdvsypbpr.hh ImgIOdvsrgb.hh GrabfileCommon.hh \
 GrabfileReader.hh GrabfileReaderV1.hh GrabfileWriter.hh GrabfileWriterV1.hh \
 LegacyGrabfileReader.hh NewGrabfileReader.hh NewGrabfileWriter.hh \
 dvsFormat.hh Utilities.hh

SOURCES = ImgIOdvsypbpr.cc ImgIOdvsrgb.cc dvsFormat.cc GrabfileReader.cc \
 GrabfileReaderV1.cc GrabfileWriterV1.cc LegacyGrabfileReader.cc \
 NewGrabfileReader.cc NewGrabfileWriter.cc Utilities.cc

PLIB = RavlRawVidIO

#LIBDEBS=DVSFileFormat

#EXAMPLES =

USESLIBS = RavlImageIO RavlOSIO RavlImage RavlExtImgIO RavlIO

PROGLIBS = DynLink

MUSTLINK=RavlRawVidIO.cc
#MAINS= readgrab.cc

EHT=RawVid.html
