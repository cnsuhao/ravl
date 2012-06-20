# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2010-12, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
PACKAGE = Ravl

DESCRIPTION= LAPACK hooks for linear algebra 

REQUIRES = LAPACK

SOURCES = LAHooksLAPACK.cc

MUSTLINK = linkLAHooksLAPACK.cc

PLIB=RavlLapack

USESLIBS = RavlMath RavlLapackWraps

PROGLIBS=RavlOS

TESTEXES=testMatrixLapack.cc
