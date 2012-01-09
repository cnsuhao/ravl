# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2009, OmniPerception Ltd
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! author=Charles Galambos
#! docentry=Ravl.API.Core.IO.XMLFactory

PACKAGE=Ravl

HEADERS=Service.hh ServiceList.hh ServiceThread.hh

SOURCES=Service.cc ServiceList.cc ServiceThread.cc

MUSTLINK=LinkRavlService.cc

PLIB= RavlService

USESLIBS=RavlXMLFactory RavlCore RavlIO RavlThreads RavlOS

PROGLIBS=

