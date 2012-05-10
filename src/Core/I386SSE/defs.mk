# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Core/Base/defs.mk"

PACKAGE = Ravl

DESCRIPTION= SSE2 versions of core operations

REQUIRES = USE_SSE2

SOURCES = VectorUtilsI386SSE.cc

MUSTLINK = linkVectorUtilsI386SSE.cc

USERCPPFLAGS=-msse2

PLIB=RavlCore

USESLIBS = RavlCore 

PROGLIBS= RavlMath RavlOS

LIBDEPS = RavlI386SSE.def

TESTEXES= testI386SSE.cc
