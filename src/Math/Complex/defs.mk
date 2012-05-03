# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Math/LinearAlgebra/defs.mk"

DESCRIPTION= Complex numbers

PACKAGE = Ravl

HEADERS = Complex.hh 

SOURCES = Complex.cc 

LIBDEPS= Complex.def

PLIB= RavlMath

USESLIBS = RavlMath

EXAMPLES = exComplx.cc
