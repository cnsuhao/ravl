# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"

PACKAGE = Ravl/Image
DESCRIPTION = Digital Video Classes
SUPPORT_ONLY = linux  
REQUIRES = libDV libavc1394

MAINS=doDvDisplay.cc doDvGrab.cc
HEADERS=PalFrame.hh ImgIODv.hh DvDecode.hh DvIFormat.hh DvDevice.hh WavFile.hh 
SOURCES=PalFrame.cc ImgIODv.cc DvDecode.cc DvIFormat.cc DvDevice.cc WavFile.cc DvFrameConvert.cc

PLIB=RavlDV

AUXDIR=lib/RAVL/libdep
AUXFILES= LibDV.def

USESLIBS=system RavlCore RavlImage RavlIO RavlOS RavlOSIO RavlDPDisplay glib GTK LibDV  RavlVideoIO
PROGLIBS= RavlDPDisplay.opt

MUSTLINK = RAVLVidIDV.cc
