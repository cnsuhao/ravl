# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
# $Id$
#! rcsid="$Id$"

PACKAGE=Ravl/Image

HEADERS=ImageRectangle.hh Image.hh RGBValue.hh RGBAValue.hh YUVValue.hh \
 ByteRGBValue.hh ByteYUVValue.hh ByteRGBAValue.hh IAValue.hh ByteIAValue.hh \
 RealRGBValue.hh RealYUVValue.hh

SOURCES=ImageRectangle.cc Image.cc ByteRGBValue.cc ByteYUVValue.cc RealRGBValue.cc \
 RealYUVValue.cc

PLIB=RavlImage

USESLIBS=RavlCore

TESTEXES=testImage.cc
