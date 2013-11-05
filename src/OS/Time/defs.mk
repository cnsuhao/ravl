# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/OS/Time/defs.mk"

PACKAGE=Ravl/OS

ANSIFLAG=

HEADERS=Date.hh DeadLineTimer.hh DateRange.hh

SOURCES=Date.cc DateIO.cc DeadLineTimer.cc DateRange.cc

PLIB=RavlOS

SUMMARY_LIB=Ravl

USESLIBS=RavlCore 

TESTEXES= testDate.cc testDeadLineTimer.cc

EHT=Ravl.API.OS.Time.eht


