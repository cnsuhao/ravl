# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2012, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#

PACKAGE=Ravl/Image

DESCRIPTION = External Image IO core

DONOT_SUPPORT=cygwin arm

PLIB = RavlExtImgIO

HEADERS = ExtImgIO.hh 

MUSTLINK =  ExtImgIO.cc

# Set what functionality the MUSTLINK module needs to initialise
ifeq (JPEG, $(filter JPEG,$(RESOURCES)))
  CFLAGS+=-DMUSTLINK_JPEG
  CCFLAGS+=-DMUSTLINK_JPEG
endif

ifeq (libPNG, $(filter libPNG,$(RESOURCES)))
  CFLAGS+=-DMUSTLINK_PNG
  CCFLAGS+=-DMUSTLINK_PNG
endif

ifeq (LibTIFF, $(filter LibTIFF,$(RESOURCES)))
  CFLAGS+=-DMUSTLINK_TIFF
  CCFLAGS+=-DMUSTLINK_TIFF
endif

EXAMPLES = exExtImgIO.cc exImgMemIO.cc

USESLIBS = RavlOSIO RavlImage RavlIO RavlImageIO

PROGLIBS = RavlDPDisplay.opt

TESTEXES=testExtImgIO.cc

SOURCES = Dummy.cc
