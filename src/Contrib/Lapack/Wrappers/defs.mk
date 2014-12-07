# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

PACKAGE      = Ravl/Lapack

DESCRIPTION  = RAVL wrapper around Lapack library

REQUIRES = LAPACK
#SUPPORT_ONLY = linux

#MAINS        = simpletest.cc

HEADERS = ev_c.hh blas2.hh blas2_c.hh lapack.hh

SOURCES = ev_c.cc blas2.cc  blas2_c.cc lapack.cc

PLIB = RavlLapackWraps

SUMMARY_LIB=Ravl

USESLIBS = LibLapack

PROGLIBS = RavlCore  RavlOS RavlIO RavlMath

EXTERNALLIBS = LibLapack.def

TESTEXES = testLapack.cc

EHT= Ravl.API.Math.Linear_Algebra.Lapack.html

