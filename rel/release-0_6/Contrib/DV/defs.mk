# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"

PACKAGE = Ravl/Image

DESCRIPTION = Digital Video Classes

#MAINS=doDvControl.cc doDvDecode.cc doDv2Wav.cc doDvGrab.cc doDv2Mpeg.cc doDvDisplay.cc 
#HEADERS=DvDevice.hh ImgIORawDv.hh WavFile.hh ImgIODvDevice.hh
#SOURCES=DvDevice.cc IPDVConvert.cc  ImgIORawDv.cc WavFile.cc ImgIODvDevice.cc

MAINS=doDvDisplay.cc doDvGrab.cc
HEADERS=PalFrame.hh ImgIODv.hh DvDecode.hh DvIFormat.hh DvDevice.hh WavFile.hh 
SOURCES=PalFrame.cc ImgIODv.cc DvDecode.cc DvIFormat.cc DvDevice.cc WavFile.cc

PLIB=RavlDV

AUXDIR=lib/RAVL/libdep
AUXFILES= LibDV.def

USESLIBS=system RavlCore RavlImage RavlIO RavlOS RavlOSIO RavlDPDisplay glib GTK LibDV 
PROGLIBS= 

MUSTLINK = RAVLVidIDV.cc
