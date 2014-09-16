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

# Check which resources that can be used by this library are available
EXTIMGIO_RESOURCES = $(filter JPEG libPNG LibTIFF, $(RESOURCES))

# Set what functionality the MUSTLINK module needs to initialise
EXTIMGIO_MUSTLINK_DEFS = $(patsubst LibTIFF,  -DMUSTLINK_TIFF , \
                          $(patsubst libPNG,  -DMUSTLINK_PNG ,\
                              $(patsubst JPEG, -DMUSTLINK_JPEG , \
                                  $(EXTIMGIO_RESOURCES))))

CFLAGS += $(EXTIMGIO_MUSTLINK_DEFS)
CCFLAGS += $(EXTIMGIO_MUSTLINK_DEFS)

# Set what external libraries we will be linking this library with
# (The JPEG and PNG RESOURCE name do not match the name of the
# USESLIBS setting, so we have to alter the name. TIFF shares
# the same name between the two settings so needs no processing)
EXTIMGIO_EXTERNALS = $(patsubst libPNG,  LibPNG, \
                      $(patsubst JPEG, LibJPEG , $(EXTIMGIO_RESOURCES)))

USESLIBS = RavlImage RavlIO RavlImageIO $(EXTIMGIO_EXTERNALS)

EXAMPLES = exExtImgIO.cc exImgMemIO.cc

PROGLIBS = RavlDPDisplay.opt

TESTEXES=testExtImgIO.cc

SOURCES = Dummy.cc
