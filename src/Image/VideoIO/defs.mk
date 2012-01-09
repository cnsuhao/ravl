# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-12, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Image/VideoIO/defs.mk"

PACKAGE = Ravl/Image

DESCRIPTION = Video IO classes.

HEADERS = ImgIOCif.hh CifFormat.hh \
          ImgIOyuv.hh yuvFormat.hh syuvFormat.hh \
          ImgIOrgb.hh rgbFormat.hh rawFormat.hh \
	  ImgIOjs.hh jsFormat.hh \
	  VidIO.hh \
	  MultiVidIO.hh DVFrame.hh AviStream.hh \
          BGRFrmIOAvi.hh DVFrmIOAvi.hh AviFormat.hh

#          VidIO.hh SubSamp2Intl.hh SubSample.hh 

SOURCES = ImgIOCif.cc CifFormat.cc \
          ImgIOyuv.cc yuvFormat.cc syuvFormat.cc \
          ImgIOrgb.cc rgbFormat.cc rawFormat.cc \
	  ImgIOjs.cc jsFormat.cc \
	  MultiVidIO.cc DVFrame.cc AviIStream.cc \
          AviOStream.cc BGRFrmIOAvi.cc DVFrmIOAvi.cc \
	  AviFormat.cc RavlVidIO.cc

#              SubSamp2Intl.cc  ImgComposites.cc


MUSTLINK = linkRavlVidIO.cc

PLIB = RavlVideoIO

EXAMPLES = exImgSeq.cc

USESLIBS = RavlImageIO RavlOSIO

PROGLIBS= RavlDPDisplay.opt DynLink

EHT=Ravl.API.Images.Video.html Ravl.API.Images.Video.Video_IO.html
