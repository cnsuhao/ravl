# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

PACKAGE = Ravl/Image

DESCRIPTION = FireWire DV Device handling.

SUPPORT_ONLY = linux linux64

REQUIRES = libDV libavc1394

MAINS=doDvGrab.cc

HEADERS= DvDevice.hh WavFile.hh PalFrame.hh

SOURCES= DvDevice.cc WavFile.cc PalFrame.cc

PLIB=RavlFireWire

USESLIBS=RavlDV LibFireWire

AUXDIR=lib$(PROJECT_DIR)/libdep

AUXFILES= LibFireWire.def

PROGLIBS= RavlDPDisplay.opt

