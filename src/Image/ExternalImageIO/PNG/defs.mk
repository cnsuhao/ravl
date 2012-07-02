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

LIBDEPS = RavlPNG.def

HEADERS = ImgIOPNG.hh ImgIOPNGB.hh PNGFormat.hh

SOURCES = ImgIOPNG.cc PNGFormat.cc

USESLIBS = RavlImage RavlIO RavlImageIO LibPNG
