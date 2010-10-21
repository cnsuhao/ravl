# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id: defs.mk 7842 2010-09-24 08:35:20Z alexkostin $"
#! file="Ravl/OS/Misc/defs.mk"

PACKAGE=Ravl/OS

DESCRIPTION=Interfaces for serial devices

HEADERS=SerialAbstract.hh SerialDirect.hh SerialDFormat.hh \
        SerialIO.hh SerialAbstractPort.hh

SOURCES=SerialAbstract.cc SerialDirect.cc SerialDFormat.cc \
        SerialIO.cc

MUSTLINK=linkSerialDirect.cc

PLIB=RavlSerialIO

USESLIBS=RavlCore RavlIO RavlOS
