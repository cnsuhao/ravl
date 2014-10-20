# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE = Ravl/Image

DONOTSUPPORT=VCPP

REQUIRES=libmpeg

DESCRIPTION = MPEG video IO classes.

HEADERS = ImgIMPEG.hh MPEGIFormat.hh
#ImgIOMpeg.hh ImgIOMpegB.hh MpegFormat.hh \
 #         ExtVidIO.hh

SOURCES = ImgIMPEG.cc MPEGIFormat.cc
#ImgIOMpeg.cc MpegFormat.cc

MUSTLINK = RAVLVidIMPEG.cc

PLIB = RavlMPEG

EXAMPLES = exMPEGI.cc

USESLIBS = RavlImageIO RavlImage RavlOSIO LibMPEG 

PROGLIBS = RavlDPDisplay.opt

EXTERNALLIBS=LibMPEG.def
#USERCFLAGS = -g
#PROGLIBS = Mopt
