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

SUMMARY_LIB=Ravl

HEADERS = ExtImgIO.hh 

MUSTLINK =  ExtImgIO.cc

# Set what functionality the MUSTLINK module needs to initialise
LOCAL_MUSTLINK_DEFS = $(patsubst LibTIFF,  -DMUSTLINK_TIFF , \
                          $(patsubst libPNG,  -DMUSTLINK_PNG ,\
                              $(patsubst JPEG, -DMUSTLINK_JPEG , \
                                  $(filter JPEG libPNG LibTIFF, $(RESOURCES)))))

CFLAGS += $(LOCAL_MUSTLINK_DEFS)
CCFLAGS += $(LOCAL_MUSTLINK_DEFS)

EXAMPLES = exExtImgIO.cc exImgMemIO.cc

USESLIBS = RavlImage RavlIO RavlImageIO LibJPEG LibPNG LibTIFF

PROGLIBS = RavlDPDisplay.opt

TESTEXES=testExtImgIO.cc

SOURCES = Dummy.cc
