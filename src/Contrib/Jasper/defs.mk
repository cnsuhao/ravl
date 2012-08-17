# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2005-11, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

ORGANISATION=OmniPerception Ltd.

DONOT_SUPPORT=

REQUIRES=LibJasper

PACKAGE=Ravl/Image

HEADERS=ImgIOJasper.hh JasperFormat.hh CompressedImageJ2k.hh JasperIF.hh

SOURCES=ImgIOJasper.cc JasperFormat.cc CompressedImageJ2k.cc

MUSTLINK=JasperImgIO.cc

PLIB=RavlImgIOJasper

USESLIBS=RavlOSIO RavlImage RavlIO

INCLUDES += $(LibJasper_CFLAGS)

PROGLIBS=RavlDPDisplay.opt

AUXDIR=lib$(PROJECT_DIR)/libdep

AUXFILES=LibJasper.def

TESTEXES=testImgIOJasper.cc
