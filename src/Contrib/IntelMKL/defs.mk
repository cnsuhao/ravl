# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2004, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Contrib/IntelMKL/defs.mk"

REQUIRES=libmkl

DONOT_SUPPORT=VCPP

PACKAGE=Ravl

HEADERS=IntelMKL.hh IntelFFT2d.hh

SOURCES=IntelMKL.cc IntelFFT2d.cc

TESTEXES= testIntelFFT2d.cc testIntelMKL.cc

PLIB=RavlIntelMKL

USESLIBS=IntelMKL RavlMath

PROGLIBS=RavlOS

AUXFILES=IntelMKL.def

AUXDIR=lib/RAVL/libdep

MUSTLINK=linkIntelMKL.cc

EHT= Ravl.API.Math.Linear_Algebra.IntelMKL.html