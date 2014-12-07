# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2010-14, OmniPerception Ltd
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

REQUIRES=iksemel RLog

PACKAGE=Ravl/XMPP

HEADERS=IksemelConnection.hh

SOURCES=IksemelConnection.cc

PLIB=RavlXMPPIksemel

SUMMARY_LIB=Ravl

MAINS= testIksemel.cc

USESLIBS=RavlOS iksemel RavlXMPP RavlRLog RLog
# including Ravl/RLog.hh in the source causes a dependency on both RavlRLog
# and RLog itself.

PROGLIBS=RavlIO RavlXMLFactory

MUSTLINK=LinkRavlXMPPIksemel.cc

EXTERNALLIBS=iksemel.def

CCPPFLAGS += -DRLOG_COMPONENT=Ravl
