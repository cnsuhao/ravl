# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2012, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#

PACKAGE=Ravl/Image

DESCRIPTION = External Image TIFF IO routines

REQUIRES=LibTIFF

DONOT_SUPPORT=cygwin arm

PLIB = RavlExtImgIO

LIBDEPS = RavlTIFF.def

HEADERS = ImgIOTiff.hh ImgIOTiffB.hh TiffFormat.hh

SOURCES = ImgIOTiff.cc TiffFormat.cc 

USESLIBS = RavlImage RavlIO RavlImageIO LibTIFF
