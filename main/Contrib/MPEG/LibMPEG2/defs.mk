# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2003, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"


PACKAGE=Ravl/Image

HEADERS= ImgIOMPEG2.hh LibMPEG2Format.hh

SOURCES= ImgIOMPEG2.cc LibMPEG2Format.cc

PLIB=RavlLibMPEG2

MUSTLINK= MustLinkLibMPEG2.cc

USESLIBS=RavlImage RavlIO LibMPEG2 RavlDPDisplay

EXAMPLES=testMPEG2.cc

AUXDIR=lib/RAVL/libdep

AUXFILES=LibMPEG2.def
