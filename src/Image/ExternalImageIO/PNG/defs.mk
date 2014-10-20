# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2012, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#

PACKAGE=Ravl/Image

DESCRIPTION = External Image PNG IO routines

REQUIRES=libPNG

DONOT_SUPPORT=cygwin arm

PLIB = RavlExtImgIO

SUMMARY_LIB=Ravl

LIBDEPS = RavlPNG.def

HEADERS = ImgIOPNG.hh ImgIOPNGB.hh PNGFormat.hh

SOURCES = ImgIOPNG.cc PNGFormat.cc

# Check which resources that can be used by this library are available
EXTIMGIO_RESOURCES = $(filter JPEG libPNG LibTIFF, $(RESOURCES))

# Set what external libraries we will be linking this library with
# (The JPEG and PNG RESOURCE name do not match the name of the
# USESLIBS setting, so we have to alter the name. TIFF shares
# the same name between the two settings so needs no processing)
EXTIMGIO_EXTERNALS = $(patsubst libPNG,  LibPNG, \
                      $(patsubst JPEG, LibJPEG , $(EXTIMGIO_RESOURCES)))

USESLIBS = RavlImage RavlIO RavlImageIO $(EXTIMGIO_EXTERNALS)
