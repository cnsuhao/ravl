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

EXAMPLES = exExtImgIO.cc exImgMemIO.cc

USESLIBS = RavlOSIO RavlImage RavlIO RavlImageIO

PROGLIBS = RavlDPDisplay.opt

TESTEXES=testExtImgIO.cc

SOURCES = Dummy.cc
